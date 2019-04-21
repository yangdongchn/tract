use super::PaddingSpec;
use crate::internal::*;
use ndarray::prelude::*;
#[cfg(not(debug_assertions))]
use no_panic::no_panic;

use num_traits::AsPrimitive;

use itertools::Itertools;
use std::ops::Range;

fn storage_strides(shape: &[usize], inner_stride: usize) -> TVec<isize> {
    let mut strides: TVec<isize> = tvec![inner_stride as isize];
    for dim in shape.iter().skip(1).rev() {
        let previous = strides.last().unwrap();
        strides.push(*dim as isize * previous);
    }
    strides.reverse();
    strides
}

fn offset(
    coords: impl IntoIterator<Item = impl AsPrimitive<isize>>,
    strides: &[impl AsPrimitive<isize>],
) -> isize {
    coords.into_iter().zip(strides).map(|(c, s)| c.as_() * s.as_()).sum::<isize>()
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatchSpec {
    pub input_shape: TVec<usize>,
    pub kernel_shape: TVec<usize>,
    pub dilations: TVec<usize>,
    pub strides: TVec<usize>,
    pub padding: PaddingSpec,
    pub input_storage_stride: usize,
    pub output_storage_stride: usize,
}

impl PatchSpec {
    pub fn into_patch(self) -> Patch {
        let computed_padding = self.padding.compute(
            &*self.input_shape,
            &self.kernel_shape,
            &*self.dilations,
            &*self.strides,
        );
        let pad_before: TVec<usize> = computed_padding.dims.iter().map(|d| d.pad_before).collect();
        let pad_after: TVec<usize> = computed_padding.dims.iter().map(|d| d.pad_after).collect();
        let output_shape: TVec<usize> = computed_padding.dims.iter().map(|d| d.output).collect();

        let data_field: Vec<isize> = ::ndarray::indices(&*self.kernel_shape)
            .into_iter()
            .flat_map(|coords| {
                coords
                    .slice()
                    .to_vec()
                    .into_iter()
                    .enumerate()
                    .map(|(ix, c)| (c * self.dilations[ix]) as isize - pad_before[ix] as isize)
            })
            .collect();
        let data_field = Array2::from_shape_vec(
            (self.kernel_shape.iter().cloned().product(), self.kernel_shape.len()),
            data_field,
        )
        .unwrap();
        let data_field_min_max: TVec<_> = data_field
            .gencolumns()
            .into_iter()
            .map(|col| (col.iter().min().cloned().unwrap(), col.iter().max().cloned().unwrap()))
            .collect();

        let input_storage_strides = storage_strides(&*self.input_shape, self.input_storage_stride);
        let output_storage_strides = storage_strides(&*output_shape, self.output_storage_stride);

        let standard_layout_data_field: Vec<isize> = data_field
            .outer_iter()
            .map(|coords| offset(coords.iter().cloned(), &*input_storage_strides))
            .collect();

        let sliding_storage_strides: TVec<isize> = input_storage_strides
            .iter()
            .zip(self.strides.iter())
            .map(|(&a, &b)| a * b as isize)
            .collect();

        let mut zones = vec![];
        for ix in 0..self.input_shape.len() {
            let axis = PatchAxis::new(
                self.input_shape[ix],
                self.kernel_shape[ix],
                pad_before[ix],
                pad_after[ix],
                output_shape[ix],
                self.strides[ix],
                self.dilations[ix],
            );
            let axis_zones = axis.zones();
            if ix == 0 {
                zones.extend(axis_zones.into_iter().map(|x| tvec!(x)));
            } else {
                let previous: Vec<_> = zones.drain(..).collect();
                for z1 in previous {
                    for z2 in &axis_zones {
                        let mut z1 = z1.clone();
                        z1.push(z2.clone());
                        zones.push(z1);
                    }
                }
            }
        }

        let mut zones: Vec<Zone> = zones
            .into_iter()
            .map(|zone| {
                let valid = zone.iter().all(|axis| axis.1 == None);
                let output_shape = zone.iter().map(|axis| axis.0.end - axis.0.start).collect();
                let output_ranges = zone.iter().map(|axis| axis.0.clone()).collect();
                let output_zone_offset =
                    offset(zone.iter().map(|axis| axis.0.start), &*output_storage_strides);
                let input_zone_offset =
                    offset(zone.iter().map(|axis| axis.0.start), &*self.strides);
                let window_offsets =
                    ndarray::indices(&*self.kernel_shape)
                        .into_iter()
                        .zip(standard_layout_data_field.iter())
                        .enumerate()
                        .filter(|(_ix, (coords, _offset))| {
                            coords.slice().iter().zip(zone.iter()).all(|(&x, axis)| {
                                !axis.1.as_ref().map(|mask| mask[x]).unwrap_or(false)
                            })
                        })
                        .map(|(ix, (_coords, &window_offset))| (ix, window_offset))
                        .collect();
                Zone {
                    valid,
                    output_shape,
                    output_ranges,
                    input_zone_offset,
                    output_zone_offset,
                    window_offsets,
                }
            })
            .collect();

        zones.sort_by_key(|z| !z.valid as usize);

        Patch {
            spec: self,
            padded: pad_before.iter().any(|&p| p != 0) || pad_after.iter().any(|&p| p != 0),
            pad_before,
            pad_after,
            sliding_storage_strides,
            output_shape,
            data_field,
            data_field_min_max,
            standard_layout_data_field,
            zones,
            output_storage_strides,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Patch {
    pub spec: PatchSpec,
    pub pad_before: TVec<usize>,
    pub pad_after: TVec<usize>,
    pub padded: bool,
    /// geometric strides * storage strides in input
    pub sliding_storage_strides: TVec<isize>,
    /// output storage strides
    pub output_storage_strides: TVec<isize>,
    pub output_shape: TVec<usize>,
    pub data_field: Array2<isize>,
    pub data_field_min_max: TVec<(isize, isize)>,
    pub standard_layout_data_field: Vec<isize>,
    pub zones: Vec<Zone>,
}

impl Patch {
    #[inline]
    pub fn rank(&self) -> usize {
        self.spec.input_shape.len()
    }

    /*
    pub fn output_full_shape(&self, channels: usize) -> TVec<usize> {
        let mut v = self.input_shape.shape.clone();
        v[self.input_shape.c_axis()] = channels;
        v[self.input_shape.hw_axes()].copy_from_slice(&self.output_shape);
        v
    }
    */

    unsafe fn is_valid(&self, spatial_coords: &[usize]) -> bool {
        for ix in 0..self.rank() {
            let c = *spatial_coords.get_unchecked(ix) as isize;
            let strides = *self.spec.strides.get_unchecked(ix) as isize;
            let pos = c * strides;
            let min_max = self.data_field_min_max.get_unchecked(ix);
            if pos + min_max.0 < 0
                || pos + min_max.1 >= *self.spec.input_shape.get_unchecked(ix) as isize
            {
                return false;
            }
        }
        true
    }

    /*
    pub fn visit_zone_1<'p>(
        &'p self,
        zone: &'p [Range<usize>],
        valid_hint: Option<bool>,
    ) -> impl Iterator<Item = (usize, Option<bool>)> + 'p {
        let shape = zone[0].end - zone[0].start;
        ndarray::indices(shape)
            .into_iter()
            .map(move |coords| (unsafe { zone.get_unchecked(0).start + coords }, valid_hint))
    }

    pub fn visit_zone_2<'p>(
        &'p self,
        zone: &'p [Range<usize>],
        valid_hint: Option<bool>,
    ) -> impl Iterator<Item = ((usize, usize), Option<bool>)> + 'p {
        let shape = (zone[0].end - zone[0].start, zone[1].end - zone[1].start);
        ndarray::indices(shape).into_iter().map(move |coords| {
            (
                unsafe {
                    (zone.get_unchecked(0).start + coords.0, zone.get_unchecked(1).start + coords.1)
                },
                valid_hint,
            )
        })
    }

    pub fn visit_zone_d<'p>(
        &'p self,
        zone: &'p [Range<usize>],
        valid_hint: Option<bool>,
    ) -> impl Iterator<Item = (TVec<usize>, Option<bool>)> + 'p {
        let shape: Vec<usize> = zone.iter().map(|z| z.end - z.start).collect();
        ndarray::indices(shape).into_iter().map(move |coords| {
            let mut coords: TVec<usize> = coords.slice().into();
            for i in 0..coords.len() {
                coords[i] += zone[i].start;
            }
            (coords, valid_hint)
        })
    }

    pub fn visit_all_1(&self) -> impl Iterator<Item = (usize, Option<bool>)> + '_ {
        self.visit_valid_1().chain(self.visit_invalid_1())
    }

    pub fn visit_valid_1(&self) -> impl Iterator<Item = (usize, Option<bool>)> + '_ {
        self.visit_zone_1(&*self.valid_output_zone, Some(true))
    }

    pub fn visit_invalid_1(&self) -> impl Iterator<Item = (usize, Option<bool>)> + '_ {
        self.invalid_output_zones.iter().flat_map(move |z| self.visit_zone_1(z, Some(false)))
    }

    pub fn visit_all_2(&self) -> impl Iterator<Item = Point> + '_ {
        self.visit_valid_2().chain(self.visit_invalid_2())
    }

    pub fn visit_valid_2(&self) -> impl Iterator<Item = ((usize, usize), Option<bool>)> + '_ {
        self.visit_zone_2(&*self.valid_output_zone, Some(true))
    }

    pub fn visit_invalid_2(&self) -> impl Iterator<Item = ((usize, usize), Option<bool>)> + '_ {
        self.invalid_output_zones.iter().flat_map(move |z| self.visit_zone_2(z, Some(false)))
    }

    pub fn visit_all_d(&self) -> impl Iterator<Item = (TVec<usize>, Option<bool>)> + '_ {
        self.visit_valid_d().chain(self.visit_invalid_d())
    }

    pub fn visit_valid_d(&self) -> impl Iterator<Item = (TVec<usize>, Option<bool>)> + '_ {
        self.visit_zone_d(&*self.valid_output_zone, Some(true))
    }

    pub fn visit_invalid_d(&self) -> impl Iterator<Item = (TVec<usize>, Option<bool>)> + '_ {
        self.invalid_output_zones.iter().flat_map(move |z| self.visit_zone_d(z, Some(false)))
    }
    */

    pub fn at<'p>(&'p self, coords: &[usize]) -> PatchIterator<'p> {
        self.at_hint(coords, None)
    }

    pub fn at_hint<'p>(&'p self, coords: &[usize], hint: Option<bool>) -> PatchIterator<'p> {
        unsafe {
            let mut center = 0;
            for i in 0..self.sliding_storage_strides.len() {
                center += *self.sliding_storage_strides.get_unchecked(i)
                    * *coords.get_unchecked(i) as isize;
            }
            let valid = hint.unwrap_or_else(|| !self.padded || self.is_valid(coords));
            if valid {
                PatchIterator::Fast(FastPatchIterator { patch: &self, center, item: 0 })
            } else {
                let mut input_patch_center: TVec<_> = coords.into();
                input_patch_center
                    .iter_mut()
                    .zip(self.spec.strides.iter())
                    .for_each(|(a, &b)| *a *= b as usize);
                PatchIterator::Safe(SafePatchIterator {
                    patch: self,
                    item: 0,
                    input_patch_center,
                    center,
                })
            }
        }
    }

    pub fn zones(&self) -> impl Iterator<Item = ZoneVisitor> {
        (0..self.zones.len()).map(move |zone_id| ZoneVisitor { patch: &self, zone_id })
    }

    pub fn valid_zone(&self) -> Option<ZoneVisitor> {
        if self.zones[0].valid {
            Some(ZoneVisitor { patch: &self, zone_id: 0 })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Zone {
    valid: bool,
    input_zone_offset: isize,
    output_zone_offset: isize,
    output_ranges: TVec<Range<usize>>,
    output_shape: TVec<usize>,
    window_offsets: TVec<(usize, isize)>,
}

impl Zone {
    pub fn contains_output(&self, coords: &[usize]) -> bool {
        self.output_ranges.iter().zip(coords).all(|(range, &x)| x >= range.start && x < range.end)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ZoneVisitor<'p> {
    patch: &'p Patch,
    zone_id: usize,
}

impl<'p> ZoneVisitor<'p> {
    fn zone(&self) -> &Zone {
        unsafe { self.patch.zones.get_unchecked(self.zone_id) }
    }
    pub fn input_offset(&self) -> isize {
        self.zone().input_zone_offset
    }
    pub fn output_offset(&self) -> isize {
        self.zone().output_zone_offset
    }
    pub fn input_offsets(&self) -> impl Iterator<Item = isize> + '_ {
        let zone = self.zone();
        ndarray::indices(&*zone.output_shape).into_iter().map(move |coords| {
            offset(coords.slice().iter().cloned(), &*self.patch.sliding_storage_strides)
        })
    }
    pub fn visit(&'p self) -> impl Iterator<Item = Window<'p>> + 'p {
        let zone = self.zone();
        ndarray::indices(&*zone.output_shape).into_iter().map(move |coords| Window {
            zone,
            input_offset: offset(
                coords.slice().iter().cloned(),
                &*self.patch.sliding_storage_strides,
            ),
            output_offset: offset(
                coords.slice().iter().cloned(),
                &*self.patch.output_storage_strides,
            ),
            coords: coords.slice().into(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Window<'p> {
    zone: &'p Zone,
    input_offset: isize,
    output_offset: isize,
    coords: TVec<usize>,
}

impl<'p> Window<'p> {
    pub fn input_offset(&self) -> isize {
        self.input_offset
    }
    pub fn output_offset(&self) -> isize {
        self.output_offset
    }
    pub fn field_offsets(&self) -> impl Iterator<Item = &'p (usize, isize)> {
        self.zone.window_offsets.iter()
    }
}

#[derive(Debug)]
pub enum PatchIterator<'p> {
    Fast(FastPatchIterator<'p>),
    Safe(SafePatchIterator<'p>),
}

impl<'p> PatchIterator<'p> {
    pub fn rewind(&mut self) {
        match self {
            &mut PatchIterator::Fast(ref mut it) => it.item = 0,
            &mut PatchIterator::Safe(ref mut it) => it.item = 0,
        }
    }
}
impl<'p> Iterator for PatchIterator<'p> {
    type Item = Option<isize>;
    #[inline(always)]
    fn next(&mut self) -> Option<Option<isize>> {
        match self {
            &mut PatchIterator::Fast(ref mut it) => it.next(),
            &mut PatchIterator::Safe(ref mut it) => it.next(),
        }
    }
}

#[derive(Debug)]
pub struct FastPatchIterator<'p> {
    patch: &'p Patch,
    center: isize,
    item: usize,
}

impl<'p> Iterator for FastPatchIterator<'p> {
    type Item = Option<isize>;
    #[inline(always)]
    #[cfg_attr(not(debug_assertions), no_panic)]
    fn next(&mut self) -> Option<Option<isize>> {
        if self.item == self.patch.standard_layout_data_field.len() {
            return None;
        }
        unsafe {
            self.item += 1;
            Some(Some(*self.patch.standard_layout_data_field.get_unchecked(self.item - 1)))
        }
    }
}

#[derive(Debug)]
pub struct SafePatchIterator<'p> {
    patch: &'p Patch,
    item: usize,
    input_patch_center: TVec<usize>,
    center: isize,
}

impl<'p> Iterator for SafePatchIterator<'p> {
    type Item = Option<isize>;
    #[cfg_attr(not(debug_assertions), no_panic)]
    fn next(&mut self) -> Option<Option<isize>> {
        unsafe {
            let patch = self.patch;
            if self.item == patch.standard_layout_data_field.len() {
                return None;
            }
            let input_shape = &patch.spec.input_shape;
            let img_offset =
                patch.data_field.as_ptr().offset((self.item * self.patch.rank()) as isize);

            for ix in 0..self.patch.rank() {
                let pos = *self.input_patch_center.get_unchecked(ix) as isize
                    + *img_offset.offset(ix as isize);
                if pos < 0 || pos as usize >= *input_shape.get_unchecked(ix) {
                    self.item += 1;
                    return Some(None);
                }
            }
            self.item += 1;
            Some(Some(*self.patch.standard_layout_data_field.get_unchecked(self.item - 1)))
        }
    }
}

#[derive(Clone, Debug, new)]
struct PatchAxis {
    input_dim: usize,
    kernel_dim: usize,
    pad_before: usize,
    pad_after: usize,
    output_dim: usize,
    stride: usize,
    dilation: usize,
}

impl PatchAxis {
    fn valid_range(&self) -> Range<usize> {
        let min = self.pad_before.div_ceil(self.stride);
        let max = self.output_dim - self.pad_after.div_ceil(self.stride);
        min..max
    }

    fn invalid_at_left(&self, pos: usize) -> usize {
        let center_pos = pos * self.stride;
        self.pad_before.saturating_sub(center_pos).div_ceil(self.dilation)
    }

    fn invalid_at_right(&self, pos: usize) -> usize {
        let center_pos = pos * self.stride;
        self.pad_after.saturating_sub(self.input_dim - center_pos - 1).div_ceil(self.dilation)
    }

    fn make_invalid_zones(&self, range: Range<usize>) -> TVec<(Range<usize>, Option<TVec<bool>>)> {
        range
            .map(move |ix| (ix, (self.invalid_at_left(ix), self.invalid_at_right(ix))))
            .group_by(|&pair| pair.1)
            .into_iter()
            .map(move |(invalid, pairs)| {
                let (min, max) = pairs.map(|p| p.0).minmax().into_option().unwrap();
                let mut mask = tvec!(false; self.kernel_dim);
                for i in 0..invalid.0 {
                    mask[i] = true;
                }
                for i in 0..invalid.1 {
                    mask[self.kernel_dim - 1 - i] = true;
                }
                (min..max + 1, Some(mask))
            })
            .collect()
    }

    fn zones(&self) -> TVec<(Range<usize>, Option<TVec<bool>>)> {
        let mut zones = tvec!();
        let valid_range = self.valid_range();
        if valid_range.start > 0 {
            zones.extend(self.make_invalid_zones(0..valid_range.start));
        }
        if valid_range.start != valid_range.end {
            zones.push((valid_range.clone(), None));
        }
        if valid_range.end < self.output_dim {
            zones.extend(self.make_invalid_zones(valid_range.end..self.output_dim));
        }
        zones
    }
}

#[cfg(test)]
pub mod test {
    use super::super::DataFormat;
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;
    use proptest::*;

    // • 0 1 2 3 4 • -> 3 -> (0) 1 2 3 (4)
    fn axis_5_3() -> PatchAxis {
        PatchAxis::new(5, 3, 1, 1, 5, 1, 1)
    }

    // • • 0 1 2 3 4 • -> 4 -> (0) (1) 2 3 (4)
    fn axis_5_4() -> PatchAxis {
        PatchAxis::new(5, 4, 2, 1, 5, 1, 1)
    }

    // • • 0 1 2 3 4 • • -> 4 -> (0) (1) 2 (3) (4)
    fn axis_5_5() -> PatchAxis {
        PatchAxis::new(5, 5, 2, 2, 5, 1, 1)
    }

    // • 0 1 2 3 4 • -> 3 -> (0) 2 (4)
    fn axis_5_3_s2() -> PatchAxis {
        PatchAxis::new(5, 3, 1, 1, 3, 2, 1)
    }

    // • • 0 1 2 3 4 • • -> 3x2 -> (0) (1) 2 (3) (4)
    fn axis_5_3_d2() -> PatchAxis {
        PatchAxis::new(5, 3, 2, 2, 5, 1, 2)
    }

    // 0 1 2 3 4 5 6 7 8 9 -> 2 -> 0 3 6
    fn axis_10_2_s3_valid() -> PatchAxis {
        PatchAxis::new(10, 2, 0, 0, 3, 1, 1)
    }

    #[test]
    fn axis_valid_ranges() {
        assert_eq!(axis_5_3().valid_range(), 1..4);
        assert_eq!(axis_5_4().valid_range(), 2..4);
        assert_eq!(axis_5_5().valid_range(), 2..3);
        assert_eq!(axis_5_3_s2().valid_range(), 1..2);
        assert_eq!(axis_5_3_d2().valid_range(), 2..3);
    }

    #[test]
    fn axis_invalid_at_left() {
        assert_eq!(axis_5_3().invalid_at_left(0), 1);
        assert_eq!(axis_5_3().invalid_at_left(1), 0);
        assert_eq!(axis_5_3().invalid_at_left(2), 0);

        assert_eq!(axis_5_4().invalid_at_left(0), 2);
        assert_eq!(axis_5_4().invalid_at_left(1), 1);
        assert_eq!(axis_5_4().invalid_at_left(2), 0);

        assert_eq!(axis_5_5().invalid_at_left(0), 2);
        assert_eq!(axis_5_5().invalid_at_left(1), 1);
        assert_eq!(axis_5_5().invalid_at_left(2), 0);

        assert_eq!(axis_5_3_d2().invalid_at_left(0), 1);
        assert_eq!(axis_5_3_d2().invalid_at_left(1), 1);
        assert_eq!(axis_5_3_d2().invalid_at_left(2), 0);
    }

    #[test]
    fn axis_invalid_at_right() {
        assert_eq!(axis_5_3().invalid_at_right(0), 0);
        assert_eq!(axis_5_3().invalid_at_right(3), 0);
        assert_eq!(axis_5_3().invalid_at_right(4), 1);

        assert_eq!(axis_5_4().invalid_at_right(0), 0);
        assert_eq!(axis_5_4().invalid_at_right(3), 0);
        assert_eq!(axis_5_4().invalid_at_right(4), 1);

        assert_eq!(axis_5_5().invalid_at_right(0), 0);
        assert_eq!(axis_5_5().invalid_at_right(3), 1);
        assert_eq!(axis_5_5().invalid_at_right(4), 2);
    }

    #[test]
    fn axis_5_3_zones() {
        let zones = axis_5_3().zones();
        assert_eq!(
            zones,
            tvec!(
                (0..1, Some(tvec!(true, false, false))),
                (1..4, None),
                (4..5, Some(tvec!(false, false, true)))
            )
        );
    }

    #[test]
    fn axis_5_3_s2_zones() {
        let zones = axis_5_3_s2().zones();
        assert_eq!(
            zones,
            tvec!(
                (0..1, Some(tvec!(true, false, false))),
                (1..2, None),
                (2..3, Some(tvec!(false, false, true)))
            )
        );
    }

    #[test]
    fn axis_5_3_d2_zones() {
        let zones = axis_5_3_d2().zones();
        assert_eq!(
            zones,
            tvec!(
                (0..2, Some(tvec!(true, false, false))),
                (2..3, None),
                (3..5, Some(tvec!(false, false, true)))
            )
        );
    }

    #[test]
    fn axis_10_2_s3_valid_zones() {
        let zones = axis_10_2_s3_valid().zones();
        assert_eq!(zones, tvec!((0..3, None),));
    }

    fn field(kdim: &[usize], dilations: &[usize]) -> Array2<isize> {
        let patch = PatchSpec {
            input_shape: tvec![10; kdim.len()],
            kernel_shape: kdim.into(),
            dilations: dilations.into(),
            strides: tvec![1; kdim.len()],
            padding: PaddingSpec::Explicit(tvec![0; kdim.len()], tvec![0; kdim.len()]),
            input_storage_stride: 1,
            output_storage_stride: 1,
        }
        .into_patch();
        patch.data_field
    }

    #[test]
    #[ignore]
    fn test_field() {
        assert_eq!(field(&[3], &[1]), arr2(&[[0], [1], [2]]));
        assert_eq!(field(&[3], &[2]), arr2(&[[0], [2], [4]]));
        assert_eq!(field(&[2, 2], &[1, 1]), arr2(&[[0, 0], [0, 1], [1, 0], [1, 1]]));
        assert_eq!(field(&[2, 2], &[2, 1]), arr2(&[[0, 0], [0, 1], [2, 0], [2, 1]]));
    }

    pub fn patch_2d(output_channels: usize) -> BoxedStrategy<PatchSpec> {
        (
            Just(DataFormat::NCHW),
            (1usize..3, 1usize..3),
            1usize..3,
            (1usize..3, 1usize..3),
            //prop_oneof![PaddingSpec::SameLower, PaddingSpec::Valid],
            Just(PaddingSpec::SameLower),
            (1usize..4, 1usize..4),
        )
            .prop_flat_map(|p| {
                let size = p.3;
                (Just(p), (size.0 + 5..=size.0 + 10, size.1 + 5..=size.1 + 10))
            })
            .prop_map(move |((fmt, dil, c, ks, pad, strides), inp)| {
                let shape = fmt.shape([1, c, inp.0, inp.1]);
                PatchSpec {
                    input_shape: shape.hw_dims().into(),
                    kernel_shape: tvec!(ks.0, ks.1),
                    dilations: tvec!(dil.0, dil.1),
                    strides: tvec![strides.0, strides.1],
                    padding: pad,
                    input_storage_stride: shape.hw_stride(),
                    output_storage_stride: if fmt == DataFormat::NCHW {
                        1
                    } else {
                        output_channels
                    },
                }
            })
            .boxed()
    }

    fn check_zoning(p: Patch) -> TestCaseResult {
        let valid_zone = &p.zones[0];
        prop_assert!(valid_zone.valid);
        let invalid_zones = &p.zones[1..];
        prop_assert!(invalid_zones.iter().all(|z| !z.valid));
        for coords in ndarray::indices(&*p.output_shape) {
            let inside_valid = valid_zone.contains_output(coords.slice());
            let invalid_count =
                invalid_zones.iter().filter(|z| z.contains_output(coords.slice())).count();
            unsafe {
                prop_assert_eq!(
                    inside_valid,
                    p.is_valid(coords.slice()),
                    "valid coords {:?}, valid_zone: {:?} inside_valid: {:?} p: {:?}",
                    coords.slice(),
                    valid_zone,
                    inside_valid,
                    p
                );
            }
            if inside_valid {
                prop_assert_eq!(invalid_count, 0);
            } else {
                prop_assert_eq!(
                    invalid_count,
                    1,
                    "invalid coords {:?}, valid_zone: {:?} inside_valid: {:?} invalid_zones: {:?} p: {:?}",
                    coords.slice(),
                    valid_zone,
                    inside_valid,
                    invalid_zones,
                    p
                );
            }
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn test_zoning(p in patch_2d(1)) {
            check_zoning(p.into_patch())?;
        }

        /*
        #[test]
        fn test_zone_visitor(p in patch_2d()) {
            let mut output = ndarray::ArrayD::<i32>::zeros(&*p.output_shape);
            for (c, _v) in p.visit_all_2() {
                prop_assert!(output[[c.0, c.1]] == 0);
                output[[c.0, c.1]] = 1;
            }
            assert!(output.iter().all(|&x| x == 1));
        }
        */
    }

    #[test]
    fn test_zoning_1() {
        check_zoning(
            PatchSpec {
                dilations: tvec!(1, 1),
                kernel_shape: tvec![2, 1],
                padding: PaddingSpec::SameLower,
                strides: tvec![3, 1],
                input_shape: tvec!(10, 6),
                input_storage_stride: 1,
                output_storage_stride: 1,
            }
            .into_patch(),
        )
        .unwrap();
    }

    #[test]
    fn test_zone_visitor_1() {
        let p = PatchSpec {
            dilations: tvec!(1, 1),
            kernel_shape: tvec![2, 1],
            padding: PaddingSpec::SameLower,
            strides: tvec![1, 2],
            input_shape: tvec!(2, 2),
            input_storage_stride: 1,
            output_storage_stride: 1,
        }
        .into_patch();
        let mut output = ndarray::ArrayD::<i32>::zeros(&*p.output_shape);
        let slice = output.as_slice_mut().unwrap();
        for z in p.zones() {
            for w in z.visit() {
                let offset = (z.output_offset() + w.output_offset()) as usize;
                assert!(slice[offset] == 0);
                slice[offset] = 1;
            }
        }
        assert!(output.iter().all(|&x| x == 1));
    }
}
