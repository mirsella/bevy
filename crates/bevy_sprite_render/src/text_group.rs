//! Cached, isolated text composition shared by UI and world-space text.

use crate::SpriteAssetEvents;
use bevy_app::{App, Plugin};
use bevy_asset::{
    embedded_asset, load_embedded_asset, AssetEvent, AssetId, AssetServer, Assets, Handle,
};
use bevy_color::{Alpha, LinearRgba};
use bevy_ecs::{change_detection::Tick, entity::EntityHashMap, prelude::*, query::QueryData};
use bevy_image::Image;
use bevy_math::{ops, Rect, UVec2, Vec2};
use bevy_mesh::VertexBufferLayout;
use bevy_platform::{collections::HashMap, time::Instant};
use bevy_render::{
    diagnostic::RecordDiagnostics,
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    texture::GpuImage,
    Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_text::{
    glyph_effect_padding, FontSmoothing, Strikethrough, StrikethroughColor, TextBackgroundColor,
    TextColor, TextEffectLayer, TextEntity, TextLayoutInfo, TextPaintOrder, Underline,
    UnderlineColor,
};
use bytemuck::{Pod, Zeroable};
use core::{ops::Range, time::Duration};
use guillotiere::{size2, AllocId, AtlasAllocator};

const DEFAULT_PAGE_SIZE: u32 = 512;

pub struct TextGroupPlugin;

impl Plugin for TextGroupPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "text_group.wgsl");
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<TextGroupCache>()
                .add_systems(RenderStartup, init_pipeline)
                .add_systems(
                    Render,
                    prepare_groups.in_set(RenderSystems::PrepareResources),
                )
                .add_systems(
                    RenderGraph,
                    render_groups
                        .after(RenderGraphSystems::Begin)
                        .before(RenderGraphSystems::Render),
                );
        }
    }
}

/// Paints of a span inside an isolated text group. Opacity is deliberately absent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextGroupStyle {
    pub color: LinearRgba,
    pub background: Option<LinearRgba>,
    pub underline: Option<LinearRgba>,
    pub strikethrough: Option<LinearRgba>,
}

#[derive(QueryData)]
pub struct TextGroupStyleQuery {
    color: &'static TextColor,
    background: Option<&'static TextBackgroundColor>,
    underline: Has<Underline>,
    underline_color: Option<&'static UnderlineColor>,
    strikethrough: Has<Strikethrough>,
    strikethrough_color: Option<&'static StrikethroughColor>,
}

impl TextGroupStyleQueryItem<'_, '_> {
    pub fn style(&self) -> TextGroupStyle {
        TextGroupStyle {
            color: self.color.0.to_linear(),
            background: self.background.map(|c| c.0.to_linear()),
            underline: self.underline.then(|| {
                self.underline_color
                    .map_or(self.color.0, |c| c.0)
                    .to_linear()
            }),
            strikethrough: self.strikethrough.then(|| {
                self.strikethrough_color
                    .map_or(self.color.0, |c| c.0)
                    .to_linear()
            }),
        }
    }
}

/// Effects use raster pixels and a downward-positive Y axis, for both renderers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TextGroupEffects {
    pub shadow: Option<(LinearRgba, Vec2)>,
    pub outline: Option<(LinearRgba, f32)>,
    pub paint_order: TextPaintOrder,
    pub selected_color: Option<LinearRgba>,
    pub placeholder_color: Option<LinearRgba>,
    /// World text tints color glyphs; UI preserves their original RGB.
    pub tint_color_glyphs: bool,
}

#[derive(Clone, Copy)]
pub struct TextGroupTile {
    pub image: AssetId<Image>,
    /// Atlas coordinates, excluding the transparent sampling border.
    pub rect: Rect,
    /// Corresponding local text coordinates.
    pub bounds: Rect,
}

struct Tile {
    output: TextGroupTile,
    page: usize,
    allocation: AllocId,
    capacity: UVec2,
}

struct Page {
    image: Handle<Image>,
    allocator: AtlasAllocator,
    format: TextureFormat,
    filter: FilterMode,
}

struct Atlas {
    pages: Vec<Option<Page>>,
    page_size: u32,
}

impl Atlas {
    // Replace in place: removing a slot would shift the next tile into it without
    // checking that tile's capacity, precision or sampler.
    fn ensure_tile(
        &mut self,
        tiles: &mut Vec<Tile>,
        index: usize,
        images: &Assets<Image>,
        needed: UVec2,
        format: TextureFormat,
        filter: FilterMode,
    ) {
        if let Some(tile) = tiles.get_mut(index) {
            let page = self.pages[tile.page]
                .as_ref()
                .expect("text allocation must have a page");
            if !tile.capacity.cmpge(needed).all() || page.format != format || page.filter != filter
            {
                self.free(tile);
                *tile = self.allocate(images, needed, format, filter);
            }
        } else {
            assert_eq!(index, tiles.len());
            tiles.push(self.allocate(images, needed, format, filter));
        }
    }

    fn free(&mut self, tile: &Tile) {
        let page = self.pages[tile.page]
            .as_mut()
            .expect("text allocation must have a page");
        page.allocator.deallocate(tile.allocation);
    }

    fn allocate(
        &mut self,
        images: &Assets<Image>,
        needed: UVec2,
        format: TextureFormat,
        filter: FilterMode,
    ) -> Tile {
        // Counters can grow by a few pixels without replacing their atlas allocation.
        let capacity = ((needed + UVec2::splat(15)) / 16 * 16).min(UVec2::splat(self.page_size));
        let size = size2(capacity.x as i32, capacity.y as i32);
        let existing = self.pages.iter_mut().enumerate().find_map(|(i, page)| {
            page.as_mut()
                .filter(|p| p.format == format && p.filter == filter)
                .and_then(|p| p.allocator.allocate(size).map(|a| (i, a)))
        });
        let (index, allocation) = existing.unwrap_or_else(|| {
            let mut page = Page {
                image: images.reserve_handle(),
                allocator: AtlasAllocator::new(size2(self.page_size as i32, self.page_size as i32)),
                format,
                filter,
            };
            let allocation = page
                .allocator
                .allocate(size)
                .expect("tile must fit an empty atlas page");
            let index = self
                .pages
                .iter()
                .position(Option::is_none)
                .unwrap_or(self.pages.len());
            if index == self.pages.len() {
                self.pages.push(Some(page));
            } else {
                self.pages[index] = Some(page);
            }
            (index, allocation)
        });
        let page = self.pages[index].as_ref().unwrap();
        let origin = allocation.rectangle.min;
        let min = Vec2::new(origin.x as f32 + 1., origin.y as f32 + 1.);
        Tile {
            output: TextGroupTile {
                image: page.image.id(),
                rect: Rect { min, max: min },
                bounds: Rect::default(),
            },
            page: index,
            allocation: allocation.id,
            capacity,
        }
    }
}

struct Group {
    layout_tick: Option<Tick>,
    styles: Vec<TextGroupStyle>,
    effects: TextGroupEffects,
    quads: Vec<Quad>,
    tiles: Vec<Tile>,
    used: bool,
    dirty: bool,
}

/// Actual cache work in the most recently prepared frame.
#[derive(Default, Debug, Clone, Copy)]
pub struct TextGroupStats {
    pub groups: usize,
    pub tiles: usize,
    pub pages: usize,
    pub bytes: usize,
    pub redrawn_groups: usize,
    pub uploaded_quads: usize,
    pub style_checks: usize,
    /// CPU time checking span paints, when profiling is enabled on the cache.
    pub style_check_time: Duration,
}

/// Render-world cache. Transform, clipping and group opacity never invalidate it.
#[derive(Resource)]
pub struct TextGroupCache {
    groups: EntityHashMap<Group>,
    atlas: Atlas,
    stats: TextGroupStats,
    style_checks: usize,
    style_check_time: Option<Duration>,
}

impl Default for TextGroupCache {
    fn default() -> Self {
        Self::with_page_size(DEFAULT_PAGE_SIZE)
    }
}

impl TextGroupCache {
    /// Create an empty render-world cache with square atlas pages of this size.
    /// Install before extracting text. Sizes must be powers of two from 64 to 2048.
    pub fn with_page_size(page_size: u32) -> Self {
        assert!(
            page_size.is_power_of_two() && (64..=2048).contains(&page_size),
            "text group atlas page size must be a power of two from 64 to 2048"
        );
        Self {
            groups: Default::default(),
            atlas: Atlas {
                pages: Vec::new(),
                page_size,
            },
            stats: Default::default(),
            style_checks: 0,
            style_check_time: None,
        }
    }

    /// Enable optional CPU timing of span-paint checks. Disabled by default.
    pub fn enable_style_profiling(&mut self) {
        self.style_check_time.get_or_insert(Duration::ZERO);
    }

    pub fn stats(&self) -> TextGroupStats {
        self.stats
    }

    /// Keep an existing group through a fully transparent frame, without building
    /// unseen text. Layout and paint changes are checked when it becomes visible.
    pub fn keep_cached(&mut self, entity: Entity) {
        if let Some(group) = self.groups.get_mut(&entity) {
            group.used = true;
        }
    }

    pub fn extract(
        &mut self,
        entity: Entity,
        images: &Assets<Image>,
        layout: &Ref<TextLayoutInfo>,
        sections: &[TextEntity],
        styles: &Query<TextGroupStyleQuery>,
        effects: TextGroupEffects,
    ) -> impl Iterator<Item = TextGroupTile> + '_ {
        let group = self.groups.entry(entity).or_insert_with(|| Group {
            layout_tick: None,
            styles: Vec::new(),
            effects,
            quads: Vec::new(),
            tiles: Vec::new(),
            used: false,
            dirty: true,
        });
        group.used = true;
        let mut changed = group.layout_tick != Some(layout.last_changed())
            || group.effects != effects
            || group.styles.len() != sections.len();
        let style_timer = self
            .style_check_time
            .as_mut()
            .map(|elapsed| (Instant::now(), elapsed));
        for (index, section) in sections.iter().enumerate() {
            self.style_checks += 1;
            let Ok(style) = styles.get(section.entity) else {
                tracing::warn!(
                    "Skipping cached text {entity:?}: missing section {:?}",
                    section.entity
                );
                group.used = false;
                break;
            };
            let paint = style.style();
            if let Some(previous) = group.styles.get_mut(index) {
                changed |= *previous != paint;
                *previous = paint;
            } else {
                group.styles.push(paint);
            }
        }
        group.styles.truncate(sections.len());
        if let Some((start, elapsed)) = style_timer {
            *elapsed += start.elapsed();
        }
        if group.used && changed {
            group.layout_tick = Some(layout.last_changed());
            group.effects = effects;
            build_quads(&mut group.quads, layout, &group.styles, effects);
            let format = if group
                .quads
                .iter()
                .any(|q| q.color.red > 1. || q.color.green > 1. || q.color.blue > 1.)
            {
                TextureFormat::Rgba16Float
            } else {
                TextureFormat::Rgba8Unorm
            };
            let filter = group_filter(layout, sections);
            let bounds = group
                .quads
                .iter()
                .map(|q| q.bounds)
                .reduce(|a, b| a.union(b));
            let mut tile_count = 0;
            if let Some(bounds) = bounds {
                let min = bounds.min.floor();
                let max = bounds.max.ceil();
                assert!(
                    min.is_finite() && max.is_finite(),
                    "text group bounds must be finite"
                );
                let extent = (max - min).as_uvec2();
                // Neighboring-content gutters keep filtered tile edges seamless.
                let tile_size = self.atlas.page_size - 2;
                for y in (0..extent.y).step_by(tile_size as usize) {
                    for x in (0..extent.x).step_by(tile_size as usize) {
                        let tile_bounds = Rect {
                            min: min + Vec2::new(x as f32, y as f32),
                            max: (min + Vec2::new((x + tile_size) as f32, (y + tile_size) as f32))
                                .min(max),
                        };
                        if !group.quads.iter().any(|q| overlaps(q.bounds, tile_bounds)) {
                            continue;
                        }
                        let needed = tile_bounds.size().as_uvec2() + UVec2::splat(2);
                        self.atlas.ensure_tile(
                            &mut group.tiles,
                            tile_count,
                            images,
                            needed,
                            format,
                            filter,
                        );
                        let tile = &mut group.tiles[tile_count];
                        tile.output.bounds = tile_bounds;
                        tile.output.rect.max = tile.output.rect.min + tile_bounds.size();
                        tile_count += 1;
                    }
                }
            }
            for tile in group.tiles.drain(tile_count..) {
                self.atlas.free(&tile);
            }
            // Empty compositions are valid cache entries with no GPU work to finish.
            group.dirty = !group.tiles.is_empty();
        }
        group
            .tiles
            .iter()
            .take(if group.used { group.tiles.len() } else { 0 })
            .map(|t| t.output)
    }
}

fn group_filter(layout: &TextLayoutInfo, sections: &[TextEntity]) -> FilterMode {
    // Empty spans, including the root, do not contribute a sampling preference.
    let has_glyphs = !layout.glyphs.is_empty() || !layout.placeholder_glyphs.is_empty();
    if has_glyphs
        && layout
            .glyphs
            .iter()
            .chain(&layout.placeholder_glyphs)
            .all(|glyph| {
                sections
                    .get(glyph.section_index)
                    .is_some_and(|section| section.font_smoothing == FontSmoothing::None)
            })
    {
        FilterMode::Nearest
    } else {
        FilterMode::Linear
    }
}

fn overlaps(a: Rect, b: Rect) -> bool {
    a.min.cmplt(b.max).all() && b.min.cmplt(a.max).all()
}

#[derive(Clone, Copy)]
struct Quad {
    bounds: Rect,
    image: AssetId<Image>,
    rect: Rect,
    color: LinearRgba,
    // Matches the coverage modes in the group shader: color, alpha, outline, union, solid.
    coverage: u32,
}

fn build_quads(
    quads: &mut Vec<Quad>,
    layout: &TextLayoutInfo,
    styles: &[TextGroupStyle],
    effects: TextGroupEffects,
) {
    quads.clear();
    // A span can disappear after layout and before render extraction.
    if layout
        .glyphs
        .iter()
        .chain(&layout.placeholder_glyphs)
        .any(|glyph| glyph.section_index >= styles.len())
        || layout
            .run_geometry
            .iter()
            .any(|run| run.section_index >= styles.len())
        || (!layout.preedit_underline_rects.is_empty() && styles.is_empty())
    {
        tracing::warn!("Skipping cached text: layout references missing sections");
        return;
    }
    let solid = |bounds, color| Quad {
        bounds,
        color,
        image: AssetId::default(),
        rect: Rect::default(),
        coverage: 4,
    };
    for run in &layout.run_geometry {
        if let Some(color) = styles[run.section_index].background {
            quads.push(solid(run.bounds, color));
        }
    }
    let decorations = |quads: &mut Vec<Quad>, shadow: Option<(LinearRgba, Vec2)>| {
        for run in &layout.run_geometry {
            let style = styles[run.section_index];
            for (color, position, size) in [
                (
                    style.underline,
                    run.underline_position(),
                    run.underline_size(),
                ),
                (
                    style.strikethrough,
                    run.strikethrough_position(),
                    run.strikethrough_size(),
                ),
            ] {
                if let Some(color) = color {
                    let (color, offset) = shadow.unwrap_or((color, Vec2::ZERO));
                    quads.push(solid(
                        Rect::from_center_size(position + offset, size),
                        color,
                    ));
                }
            }
        }
    };
    if effects.shadow.is_some() {
        decorations(quads, effects.shadow);
    }
    let padding = glyph_effect_padding(
        effects.shadow.is_some() || effects.outline.is_some(),
        effects.outline.map(|(_, w)| w),
    );
    for layer in effects.paint_order.layers() {
        let (paint, offset) = match layer {
            TextEffectLayer::Shadow => {
                let Some((c, o)) = effects.shadow else {
                    continue;
                };
                (Some(c), o)
            }
            TextEffectLayer::Outline => {
                let Some((c, _)) = effects.outline else {
                    continue;
                };
                (Some(c), Vec2::ZERO)
            }
            TextEffectLayer::Fill => (None, Vec2::ZERO),
        };
        for (glyphs, placeholder) in [(&layout.glyphs, false), (&layout.placeholder_glyphs, true)] {
            for glyph in glyphs {
                let atlas = &glyph.atlas_info;
                if layer == TextEffectLayer::Outline && !atlas.is_alpha_mask {
                    continue;
                }
                let rect = if atlas.is_alpha_mask || layer == TextEffectLayer::Shadow {
                    atlas.rect.inflate(layer.glyph_padding(padding))
                } else {
                    atlas.rect
                };
                let color = paint.unwrap_or_else(|| {
                    let mut color = if placeholder {
                        effects.placeholder_color.unwrap_or(styles[0].color)
                    } else {
                        styles[glyph.section_index].color
                    };
                    if !placeholder && let Some(selected) = effects.selected_color {
                        let bounds = Rect::from_center_size(glyph.position, atlas.rect.size());
                        if layout
                            .selection_rects
                            .iter()
                            .any(|r| r.contains(bounds.min) && r.contains(bounds.max))
                        {
                            color = selected;
                        }
                    }
                    if !atlas.is_alpha_mask && !effects.tint_color_glyphs {
                        color = LinearRgba::WHITE.with_alpha(color.alpha);
                    }
                    color
                });
                let coverage = match layer {
                    TextEffectLayer::Shadow if atlas.is_alpha_mask && effects.outline.is_some() => {
                        3
                    }
                    TextEffectLayer::Shadow => 1,
                    TextEffectLayer::Outline => 2,
                    TextEffectLayer::Fill if atlas.is_alpha_mask => 1,
                    TextEffectLayer::Fill => 0,
                };
                quads.push(Quad {
                    bounds: Rect::from_center_size(glyph.position + offset, rect.size()),
                    image: atlas.texture,
                    rect,
                    color,
                    coverage,
                });
            }
        }
    }
    decorations(quads, None);
    // IME underlines share glyph coordinates and paint after the text, without a shadow.
    for &bounds in &layout.preedit_underline_rects {
        quads.push(solid(bounds, styles[0].color));
    }
    quads.retain(|q| q.color.alpha > 0. && q.bounds.width() > 0. && q.bounds.height() > 0.);
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GroupInstance {
    rect: [f32; 4],
    uv: [f32; 4],
    color: [f32; 4],
    coverage: [u32; 4],
}

struct Batch {
    image: AssetId<Image>,
    range: Range<u32>,
}
struct PageDraw {
    image: AssetId<Image>,
    clear: Range<u32>,
    batches: Range<usize>,
}

#[derive(Resource)]
struct GpuGroups {
    layout: BindGroupLayoutDescriptor,
    pipelines: [(CachedRenderPipelineId, CachedRenderPipelineId); 2],
    instances: RawBufferVec<GroupInstance>,
    bindings: HashMap<AssetId<Image>, BindGroup>,
    pages: Vec<PageDraw>,
    batches: Vec<Batch>,
    groups: Vec<Entity>,
}

fn init_pipeline(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "text_group_image",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    let shader = load_embedded_asset!(assets.as_ref(), "text_group.wgsl");
    let descriptor = |clear, format| RenderPipelineDescriptor {
        label: Some(
            if clear {
                "text_group_clear"
            } else {
                "text_group_compose"
            }
            .into(),
        ),
        layout: if clear { vec![] } else { vec![layout.clone()] },
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            buffers: vec![VertexBufferLayout::from_vertex_formats(
                VertexStepMode::Instance,
                [
                    VertexFormat::Float32x4,
                    VertexFormat::Float32x4,
                    VertexFormat::Float32x4,
                    VertexFormat::Uint32x4,
                ],
            )],
            ..Default::default()
        },
        fragment: Some(FragmentState {
            shader: shader.clone(),
            entry_point: Some(if clear { "clear" } else { "fragment" }.into()),
            targets: vec![Some(ColorTargetState {
                format,
                blend: (!clear).then_some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
            ..Default::default()
        }),
        ..Default::default()
    };
    commands.insert_resource(GpuGroups {
        pipelines: [TextureFormat::Rgba8Unorm, TextureFormat::Rgba16Float].map(|format| {
            (
                cache.queue_render_pipeline(descriptor(false, format)),
                cache.queue_render_pipeline(descriptor(true, format)),
            )
        }),
        layout,
        instances: RawBufferVec::new(BufferUsages::VERTEX),
        bindings: HashMap::default(),
        pages: Vec::new(),
        batches: Vec::new(),
        groups: Vec::new(),
    });
}

fn prepare_groups(
    mut cache: ResMut<TextGroupCache>,
    mut gpu: ResMut<GpuGroups>,
    mut images: ResMut<RenderAssets<GpuImage>>,
    events: Res<SpriteAssetEvents>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipelines: Res<PipelineCache>,
) {
    let TextGroupCache {
        groups,
        atlas,
        stats,
        style_checks,
        style_check_time,
    } = &mut *cache;
    groups.retain(|_, group| {
        if group.used {
            group.used = false;
            true
        } else {
            for tile in &group.tiles {
                atlas.free(tile);
            }
            false
        }
    });
    let Atlas { pages, page_size } = atlas;
    for slot in pages.iter_mut() {
        if slot.as_ref().is_some_and(|p| p.allocator.is_empty()) {
            let page = slot.take().unwrap();
            images.remove(page.image.id());
        }
    }
    for event in &events.images {
        if let AssetEvent::Unused { id }
        | AssetEvent::Removed { id }
        | AssetEvent::Modified { id } = event
        {
            gpu.bindings.remove(id);
        }
    }
    *stats = TextGroupStats {
        groups: groups.len(),
        tiles: groups.values().map(|g| g.tiles.len()).sum(),
        pages: pages.iter().flatten().count(),
        style_checks: core::mem::take(style_checks),
        style_check_time: style_check_time
            .as_mut()
            .map_or(Duration::ZERO, core::mem::take),
        ..Default::default()
    };
    stats.bytes = pages
        .iter()
        .flatten()
        .map(|p| {
            *page_size as usize
                * *page_size as usize
                * if p.format == TextureFormat::Rgba16Float {
                    8
                } else {
                    4
                }
        })
        .sum();
    gpu.pages.clear();
    gpu.batches.clear();
    gpu.groups.clear();
    for page in pages.iter().flatten() {
        if images.get(page.image.id()).is_none() {
            let descriptor = TextureDescriptor {
                label: Some("text_group_atlas"),
                size: Extent3d {
                    width: *page_size,
                    height: *page_size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: page.format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            };
            let texture = device.create_texture(&descriptor);
            let texture_view = texture.create_view(&Default::default());
            let sampler = device.create_sampler(&SamplerDescriptor {
                label: Some("text_group_sampler"),
                min_filter: page.filter,
                mag_filter: page.filter,
                ..Default::default()
            });
            images.insert(
                page.image.id(),
                GpuImage {
                    texture,
                    texture_view,
                    sampler,
                    texture_descriptor: descriptor,
                    texture_view_descriptor: None,
                    had_data: false,
                },
            );
            for group in groups.values_mut() {
                if group
                    .tiles
                    .iter()
                    .any(|t| t.output.image == page.image.id())
                {
                    group.dirty = true;
                }
            }
        }
    }
    for (&entity, group) in groups.iter() {
        // Asset preparation can lag extraction. Keep a dirty group pending until all sources are ready.
        if group.dirty && group.quads.iter().all(|q| images.get(q.image).is_some()) {
            gpu.groups.push(entity);
        }
    }
    stats.redrawn_groups = gpu.groups.len();
    if !groups.values().any(|g| g.dirty) {
        return;
    }
    let GpuGroups {
        instances,
        bindings,
        pages: draws,
        batches,
        groups: dirty,
        layout,
        ..
    } = &mut *gpu;
    instances.clear();
    for (page_index, page) in pages
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|p| (i, p)))
    {
        let clear_start = instances.len() as u32;
        // Clear even pending groups: never expose pixels belonging to an old allocation.
        for group in groups.values().filter(|g| g.dirty) {
            for tile in &group.tiles {
                if tile.page != page_index {
                    continue;
                }
                let min = tile.output.rect.min - Vec2::ONE;
                instances.push(instance(
                    Rect::from_corners(min, min + tile.capacity.as_vec2()),
                    LinearRgba::NONE,
                    4,
                    *page_size,
                ));
            }
        }
        let clear = clear_start..instances.len() as u32;
        if clear.is_empty() {
            continue;
        }
        let batch_start = batches.len();
        for entity in dirty.iter() {
            let group = &groups[entity];
            for tile in &group.tiles {
                if tile.page != page_index {
                    continue;
                }
                let gutter = tile.output.bounds.inflate(1.);
                let offset = tile.output.rect.min - tile.output.bounds.min;
                for quad in &group.quads {
                    let geometry = if quad.coverage == 4 {
                        quad.bounds.inflate(0.5)
                    } else {
                        quad.bounds
                    };
                    if !overlaps(geometry, gutter) {
                        continue;
                    }
                    let bounds = geometry.intersect(gutter);
                    let source = images
                        .get(quad.image)
                        .expect("prepared group sources must be available");
                    let destination = Rect {
                        min: bounds.min + offset,
                        max: bounds.max + offset,
                    };
                    let index = instances.len() as u32;
                    let mut data = instance(destination, quad.color, quad.coverage, *page_size);
                    if quad.coverage == 4 {
                        let offset = bounds.min - quad.bounds.min;
                        data.uv = [offset.x, offset.y, bounds.width(), bounds.height()];
                        data.coverage[1] = quad.bounds.width().to_bits();
                        data.coverage[2] = quad.bounds.height().to_bits();
                    } else {
                        let extent = source.size_2d().as_vec2();
                        let uv = Rect {
                            min: (quad.rect.min
                                + (bounds.min - quad.bounds.min) / quad.bounds.size()
                                    * quad.rect.size())
                                / extent,
                            max: (quad.rect.min
                                + (bounds.max - quad.bounds.min) / quad.bounds.size()
                                    * quad.rect.size())
                                / extent,
                        };
                        data.uv = [uv.min.x, uv.min.y, uv.width(), uv.height()];
                    }
                    instances.push(data);
                    if let Some(batch) = batches[batch_start..]
                        .last_mut()
                        .filter(|b| b.image == quad.image)
                    {
                        batch.range.end += 1;
                    } else {
                        batches.push(Batch {
                            image: quad.image,
                            range: index..index + 1,
                        });
                    }
                    bindings.entry(quad.image).or_insert_with(|| {
                        device.create_bind_group(
                            "text_group_source",
                            &pipelines.get_bind_group_layout(layout),
                            &BindGroupEntries::sequential((&source.texture_view, &source.sampler)),
                        )
                    });
                }
            }
        }
        draws.push(PageDraw {
            image: page.image.id(),
            clear,
            batches: batch_start..batches.len(),
        });
    }
    stats.uploaded_quads = instances.len();
    instances.write_buffer(&device, &queue);
}

fn instance(bounds: Rect, color: LinearRgba, coverage: u32, page_size: u32) -> GroupInstance {
    let min = bounds.min / page_size as f32;
    let size = bounds.size() / page_size as f32;
    let gamma = |v: f32| ops::powf(v.max(0.), 1. / 2.2);
    GroupInstance {
        rect: [min.x, min.y, size.x, size.y],
        uv: [0.; 4],
        color: [
            gamma(color.red),
            gamma(color.green),
            gamma(color.blue),
            color.alpha,
        ],
        coverage: [coverage, 0, 0, 0],
    }
}

fn render_groups(
    mut cache: ResMut<TextGroupCache>,
    gpu: Res<GpuGroups>,
    images: Res<RenderAssets<GpuImage>>,
    pipelines: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if gpu.pages.is_empty() {
        return;
    }
    // Keep the whole dirty set pending until every needed pipeline is ready.
    for page in &gpu.pages {
        let format = images.get(page.image).unwrap().texture_descriptor.format;
        let (compose, clear) = gpu.pipelines[usize::from(format == TextureFormat::Rgba16Float)];
        if pipelines.get_render_pipeline(compose).is_none()
            || pipelines.get_render_pipeline(clear).is_none()
        {
            return;
        }
    }
    let diagnostics = ctx.diagnostic_recorder();
    for (index, page) in gpu.pages.iter().enumerate() {
        let image = images
            .get(page.image)
            .expect("text group page must be prepared before rendering");
        let (compose, clear) = gpu.pipelines
            [usize::from(image.texture_descriptor.format == TextureFormat::Rgba16Float)];
        let pipeline = pipelines.get_render_pipeline(compose).unwrap();
        let clear_pipeline = pipelines.get_render_pipeline(clear).unwrap();
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("text_groups"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &image.texture_view,
                resolve_target: None,
                depth_slice: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let span = diagnostics
            .as_deref()
            .map(|d| d.pass_span(&mut pass, format!("text_groups/page_{index}")));
        pass.set_vertex_buffer(0, gpu.instances.buffer().unwrap().slice(..));
        pass.set_render_pipeline(clear_pipeline);
        pass.draw(0..6, page.clear.clone());
        pass.set_render_pipeline(pipeline);
        for batch in &gpu.batches[page.batches.clone()] {
            pass.set_bind_group(0, &gpu.bindings[&batch.image], &[]);
            pass.draw(0..6, batch.range.clone());
        }
        if let Some(span) = span {
            span.end(&mut pass);
        }
    }
    for entity in &gpu.groups {
        cache.groups.get_mut(entity).unwrap().dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::system::SystemState;

    #[test]
    fn text_paint_order_is_opt_in_and_applies_across_atlases() {
        let mut layout = glyph_layout(true);
        let mut second = layout.glyphs[0].clone();
        let images = Assets::<Image>::default();
        second.atlas_info.texture = images.reserve_handle().id();
        layout.glyphs.push(second);
        let style = TextGroupStyle {
            color: LinearRgba::WHITE,
            background: None,
            underline: None,
            strikethrough: None,
        };
        assert_eq!(TextPaintOrder::default(), TextPaintOrder::StrokeThenFill);
        let mut quads = Vec::new();
        for (paint_order, expected) in [
            (TextPaintOrder::default(), [3, 3, 2, 2, 1, 1]),
            (TextPaintOrder::FillThenStroke, [3, 3, 1, 1, 2, 2]),
            (TextPaintOrder::StrokeThenFill, [3, 3, 2, 2, 1, 1]),
        ] {
            build_quads(
                &mut quads,
                &layout,
                &[style],
                TextGroupEffects {
                    shadow: Some((LinearRgba::BLACK, Vec2::ONE)),
                    outline: Some((LinearRgba::BLACK, 0.5)),
                    paint_order,
                    ..Default::default()
                },
            );
            assert_eq!(
                quads.iter().map(|q| q.coverage).collect::<Vec<_>>(),
                expected
            );
            for pair in quads.chunks_exact(2) {
                assert_ne!(pair[0].image, pair[1].image);
            }
        }
    }
    fn extract_test_group(world: &mut World, cache: &mut TextGroupCache, entity: Entity) -> usize {
        let mut state =
            SystemState::<(Query<Ref<TextLayoutInfo>>, Query<TextGroupStyleQuery>)>::new(world);
        let (layouts, styles) = state.get(world).unwrap();
        let layout = layouts.get(entity).unwrap();
        cache
            .extract(
                entity,
                &Assets::default(),
                &layout,
                &[bevy_text::TextEntity {
                    entity,
                    depth: 0,
                    font_smoothing: FontSmoothing::AntiAliased,
                }],
                &styles,
                TextGroupEffects::default(),
            )
            .count()
    }

    #[test]
    fn empty_compositions_settle_and_refresh_when_visible() {
        for layout in [TextLayoutInfo::default(), glyph_layout(true)] {
            let mut world = World::new();
            let entity = world
                .spawn((layout, TextColor(bevy_color::Color::NONE)))
                .id();
            let mut cache = TextGroupCache::default();
            for _ in 0..3 {
                assert_eq!(extract_test_group(&mut world, &mut cache, entity), 0);
                assert!(!cache.groups[&entity].dirty, "empty text needs no GPU work");
                assert!(cache.atlas.pages.is_empty());
                world.clear_trackers();
            }

            world
                .entity_mut(entity)
                .insert((glyph_layout(true), TextColor::WHITE));
            assert_eq!(extract_test_group(&mut world, &mut cache, entity), 1);
            assert!(cache.groups[&entity].dirty);

            world.clear_trackers();
            world
                .entity_mut(entity)
                .insert(TextColor(bevy_color::Color::NONE));
            assert_eq!(extract_test_group(&mut world, &mut cache, entity), 0);
            assert!(!cache.groups[&entity].dirty);
            assert!(cache
                .atlas
                .pages
                .iter()
                .flatten()
                .all(|p| p.allocator.is_empty()));
        }
    }

    #[test]
    fn edits_while_faded_out_refresh_on_reveal() {
        let mut world = World::new();
        let entity = world.spawn((glyph_layout(true), TextColor::WHITE)).id();
        let mut cache = TextGroupCache::default();
        assert_eq!(extract_test_group(&mut world, &mut cache, entity), 1);
        cache.groups.get_mut(&entity).unwrap().dirty = false;
        world.clear_trackers();

        let color = bevy_color::Color::srgb(1., 0., 0.);
        world.get_mut::<TextColor>(entity).unwrap().0 = color;
        world.get_mut::<TextLayoutInfo>(entity).unwrap().glyphs[0]
            .position
            .x += 20.;
        for _ in 0..3 {
            cache.keep_cached(entity);
            assert!(!cache.groups[&entity].dirty);
            world.clear_trackers();
        }
        assert_eq!(extract_test_group(&mut world, &mut cache, entity), 1);
        let group = &cache.groups[&entity];
        assert!(group.dirty);
        assert_eq!(group.quads[0].color, color.to_linear());
        assert_eq!(group.tiles[0].output.bounds.min, Vec2::new(20., 0.));
    }

    #[test]
    fn preedit_underlines_compose_after_glyphs_without_shadows() {
        let mut layout = glyph_layout(true);
        let underline = Rect::from_corners(Vec2::new(0., 10.), Vec2::new(12., 12.));
        layout.preedit_underline_rects.push(underline);
        let style = TextGroupStyle {
            color: LinearRgba::WHITE,
            background: None,
            underline: None,
            strikethrough: None,
        };
        let mut quads = Vec::new();
        build_quads(
            &mut quads,
            &layout,
            &[style],
            TextGroupEffects {
                shadow: Some((LinearRgba::BLACK, Vec2::ONE)),
                outline: Some((LinearRgba::BLACK, 1.)),
                ..Default::default()
            },
        );
        assert_eq!(quads.len(), 4);
        let preedit = quads.last().unwrap();
        assert_eq!(preedit.bounds, underline);
        assert_eq!(preedit.color, style.color);
        assert_eq!(preedit.coverage, 4);
    }

    fn glyph_layout(is_alpha_mask: bool) -> TextLayoutInfo {
        TextLayoutInfo {
            glyphs: vec![bevy_text::PositionedGlyph {
                position: Vec2::splat(4.),
                atlas_info: bevy_text::GlyphAtlasInfo {
                    texture: AssetId::default(),
                    rect: Rect::from_corners(Vec2::ZERO, Vec2::splat(8.)),
                    offset: Vec2::ZERO,
                    is_alpha_mask,
                },
                section_index: 0,
                line_index: 0,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn removed_sections_do_not_render_stale_glyphs() {
        let layout = glyph_layout(true);
        let mut quads = Vec::new();
        let style = TextGroupStyle {
            color: LinearRgba::WHITE,
            background: None,
            underline: None,
            strikethrough: None,
        };
        build_quads(&mut quads, &layout, &[style], TextGroupEffects::default());
        assert_eq!(quads.len(), 1);
        build_quads(&mut quads, &layout, &[], TextGroupEffects::default());
        assert!(quads.is_empty());
    }

    #[test]
    fn color_glyph_tint_preserves_each_renderers_semantics() {
        let layout = glyph_layout(false);
        let style = TextGroupStyle {
            color: LinearRgba::new(0.25, 0.5, 0.75, 0.4),
            background: None,
            underline: None,
            strikethrough: None,
        };
        let mut quads = Vec::new();
        for tint_color_glyphs in [false, true] {
            build_quads(
                &mut quads,
                &layout,
                &[style],
                TextGroupEffects {
                    shadow: Some((LinearRgba::BLACK, Vec2::ONE)),
                    outline: Some((LinearRgba::BLACK, 1.)),
                    tint_color_glyphs,
                    ..Default::default()
                },
            );
            assert_eq!(quads.len(), 2, "color glyphs skip outlines");
            assert_eq!(quads[0].color, LinearRgba::BLACK);
            assert_eq!(
                quads[1].color,
                if tint_color_glyphs {
                    style.color
                } else {
                    LinearRgba::WHITE.with_alpha(style.color.alpha)
                }
            );
        }
    }

    #[test]
    fn empty_spans_do_not_blur_nearest_glyphs() {
        let sections =
            [FontSmoothing::AntiAliased, FontSmoothing::None].map(|font_smoothing| TextEntity {
                entity: Entity::PLACEHOLDER,
                depth: 0,
                font_smoothing,
            });
        let mut layout = glyph_layout(true);
        layout.glyphs[0].section_index = 1;
        assert_eq!(group_filter(&layout, &sections), FilterMode::Nearest);
        layout.placeholder_glyphs = core::mem::take(&mut layout.glyphs);
        assert_eq!(group_filter(&layout, &sections), FilterMode::Nearest);
        layout.glyphs = glyph_layout(true).glyphs;
        assert_eq!(group_filter(&layout, &sections), FilterMode::Linear);
        assert_eq!(
            group_filter(&TextLayoutInfo::default(), &sections),
            FilterMode::Linear
        );
    }

    #[test]
    fn resizing_a_group_keeps_every_tile_compatible() {
        for page_size in [256, 512] {
            let images = Assets::<Image>::default();
            let mut atlas = Atlas {
                pages: Vec::new(),
                page_size,
            };
            let mut tiles = Vec::new();
            for index in 0..2 {
                atlas.ensure_tile(
                    &mut tiles,
                    index,
                    &images,
                    UVec2::splat(32),
                    TextureFormat::Rgba8Unorm,
                    FilterMode::Linear,
                );
            }
            for (format, filter) in [
                (TextureFormat::Rgba8Unorm, FilterMode::Linear),
                (TextureFormat::Rgba16Float, FilterMode::Nearest),
            ] {
                for index in 0..2 {
                    atlas.ensure_tile(
                        &mut tiles,
                        index,
                        &images,
                        UVec2::splat(128),
                        format,
                        filter,
                    );
                }
                assert_eq!(tiles.len(), 2);
                for tile in &tiles {
                    assert!(tile.capacity.cmpge(UVec2::splat(128)).all());
                    let page = atlas.pages[tile.page].as_ref().unwrap();
                    assert_eq!(page.format, format);
                    assert_eq!(page.filter, filter);
                }
                assert!(
                    tiles[0].page != tiles[1].page || tiles[0].output.rect != tiles[1].output.rect
                );
            }
            for tile in tiles {
                atlas.free(&tile);
            }
            assert!(atlas
                .pages
                .iter()
                .flatten()
                .all(|page| page.allocator.is_empty()));
        }
    }
}
