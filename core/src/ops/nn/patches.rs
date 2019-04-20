use super::PaddingSpec;
use crate::internal::*;
use ndarray::prelude::*;
#[cfg(not(debug_assertions))]
use no_panic::no_panic;

use itertools::Itertools;
use std::ops::Range;

struct Point {
    valid: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Patch {
    pub dilations: TVec<usize>,
    pub kernel_spatial_shape: TVec<usize>,
    pub pad_before: TVec<usize>,
    pub pad_after: TVec<usize>,
    pub padded: bool,
    pub window_strides: TVec<usize>,
    pub window_storage_strides: TVec<isize>,
    pub input_spatial_shape: TVec<usize>,
    pub output_spatial_shape: TVec<usize>,
    pub data_field: Array2<isize>,
    pub data_field_min_max: TVec<(isize, isize)>,
    pub standard_layout_data_field: Vec<isize>,
    pub valid_output_zone: TVec<Range<usize>>,
    pub invalid_output_zones: TVec<TVec<Range<usize>>>,
}

impl Patch {
    pub fn new(
        dilations: TVec<usize>,
        kernel_spatial_shape: TVec<usize>,
        padding: &PaddingSpec,
        window_strides: TVec<usize>,
        input_spatial_shape: TVec<usize>,
        input_storage_stride: usize,
    ) -> Patch {
        let computed_padding = padding.compute(
            &*input_spatial_shape,
            &kernel_spatial_shape,
            &*dilations,
            &*window_strides,
        );
        let pad_before: TVec<usize> = computed_padding.dims.iter().map(|d| d.pad_before).collect();
        let pad_after: TVec<usize> = computed_padding.dims.iter().map(|d| d.pad_after).collect();
        let output_spatial_shape: TVec<usize> =
            computed_padding.dims.iter().map(|d| d.output).collect();

        let data_field: Vec<isize> = ::ndarray::indices(&*kernel_spatial_shape)
            .into_iter()
            .flat_map(|coords| {
                coords
                    .slice()
                    .to_vec()
                    .into_iter()
                    .enumerate()
                    .map(|(ix, c)| (c * dilations[ix]) as isize - pad_before[ix] as isize)
            })
            .collect();
        let data_field = Array2::from_shape_vec(
            (kernel_spatial_shape.iter().cloned().product(), kernel_spatial_shape.len()),
            data_field,
        )
        .unwrap();
        let data_field_min_max: TVec<_> = data_field
            .gencolumns()
            .into_iter()
            .map(|col| (col.iter().min().cloned().unwrap(), col.iter().max().cloned().unwrap()))
            .collect();

        let mut input_layout_strides: Vec<usize> = vec![input_storage_stride];
        for dim in input_spatial_shape.iter().skip(1).rev() {
            let previous = input_layout_strides.last().unwrap();
            input_layout_strides.push(dim * previous);
        }
        input_layout_strides.reverse();
        let standard_layout_data_field: Vec<isize> = data_field
            .outer_iter()
            .map(|coords| {
                coords
                    .iter()
                    .zip(input_layout_strides.iter())
                    .map(|(&a, &b)| (a as isize * b as isize))
                    .sum()
            })
            .collect();

        let mut valid_output_zone = tvec!();
        let mut invalid_output_zones = tvec!();
        let mut zones = vec!();
        for ix in 0..input_spatial_shape.len() {
            let axis = PatchAxis::new(input_spatial_shape[ix],
                                      kernel_spatial_shape[ix],
                                      pad_before[ix],
                                      pad_after[ix],
                                      output_spatial_shape[ix],
                                      window_strides[ix],
                                      dilations[ix]);
            let axis_zones = axis.zones();
            if ix == 0 {
                zones.extend(axis_zones.into_iter().map(|x| tvec!(x)));
            } else {
                let previous:Vec<_> = zones.drain(..).collect();
                for z1 in previous {
                    for z2 in &axis_zones {
                        let mut z1 = z1.clone();
                        z1.push(z2.clone());
                        zones.push(z1);
                    }
                }
            }
        }

        panic!("zones: {:?}");

        for ix in 0..input_spatial_shape.len() {
            let min_max = data_field_min_max[ix];
            let min = (-min_max.0 as usize).div_ceil(window_strides[ix]) as usize;
            let max = (input_spatial_shape[ix] - min_max.1 as usize).div_ceil(window_strides[ix])
                as usize;
            if min != 0 {
                let mut invalid = valid_output_zone.clone();
                invalid.push(0..min);
                while invalid.len() < output_spatial_shape.len() {
                    invalid.push(0..output_spatial_shape[invalid.len()])
                }
                invalid_output_zones.push(invalid);
            }
            if max < output_spatial_shape[ix] {
                let mut invalid = valid_output_zone.clone();
                invalid.push(max..output_spatial_shape[ix]);
                while invalid.len() < output_spatial_shape.len() {
                    invalid.push(0..output_spatial_shape[invalid.len()])
                }
                invalid_output_zones.push(invalid);
            }
            valid_output_zone.push(min..max)
        }

        let window_storage_strides = input_layout_strides
            .iter()
            .zip(window_strides.iter())
            .map(|(a, b)| (a * b) as isize)
            .collect();

        Patch {
            dilations,
            kernel_spatial_shape,
            padded: pad_before.iter().any(|&p| p != 0) || pad_after.iter().any(|&p| p != 0),
            pad_before,
            pad_after,
            window_strides,
            window_storage_strides,
            input_spatial_shape,
            output_spatial_shape,
            data_field,
            data_field_min_max,
            standard_layout_data_field,
            valid_output_zone,
            invalid_output_zones,
        }
    }

    #[inline]
    pub fn spatial_rank(&self) -> usize {
        self.input_spatial_shape.len()
    }

    /*
    pub fn output_full_shape(&self, channels: usize) -> TVec<usize> {
        let mut v = self.input_spatial_shape.shape.clone();
        v[self.input_spatial_shape.c_axis()] = channels;
        v[self.input_spatial_shape.hw_axes()].copy_from_slice(&self.output_spatial_shape);
        v
    }
    */

    unsafe fn is_valid(&self, spatial_coords: &[usize]) -> bool {
        for ix in 0..self.spatial_rank() {
            let c = *spatial_coords.get_unchecked(ix) as isize;
            let strides = *self.window_strides.get_unchecked(ix) as isize;
            let pos = c * strides;
            let min_max = self.data_field_min_max.get_unchecked(ix);
            if pos + min_max.0 < 0
                || pos + min_max.1 >= *self.input_spatial_shape.get_unchecked(ix) as isize
            {
                return false;
            }
        }
        true
    }

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

    pub fn at<'p>(&'p self, coords: &[usize]) -> PatchIterator<'p> {
        self.at_hint(coords, None)
    }

    pub fn at_hint<'p>(&'p self, coords: &[usize], hint: Option<bool>) -> PatchIterator<'p> {
        unsafe {
            let mut center = 0;
            for i in 0..self.window_storage_strides.len() {
                center += *self.window_storage_strides.get_unchecked(i)
                    * *coords.get_unchecked(i) as isize;
            }
            let valid = hint.unwrap_or_else(|| !self.padded || self.is_valid(coords));
            if valid {
                PatchIterator::Fast(FastPatchIterator { patch: &self, center, item: 0 })
            } else {
                let mut input_patch_center: TVec<_> = coords.into();
                input_patch_center
                    .iter_mut()
                    .zip(self.window_strides.iter())
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
            let input_spatial_shape = &patch.input_spatial_shape;
            let img_offset =
                patch.data_field.as_ptr().offset((self.item * self.patch.spatial_rank()) as isize);

            for ix in 0..self.patch.spatial_rank() {
                let pos = *self.input_patch_center.get_unchecked(ix) as isize
                    + *img_offset.offset(ix as isize);
                if pos < 0 || pos as usize >= *input_spatial_shape.get_unchecked(ix) {
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

    fn field(kdim: &[usize], dilations: &[usize]) -> Array2<isize> {
        let patch = Patch::new(
            dilations.into(),
            kdim.into(),
            &PaddingSpec::Explicit(tvec![0; kdim.len()], tvec![0; kdim.len()]),
            tvec![1; kdim.len()],
            tvec![10; kdim.len()],
            1,
        );
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

    pub fn patch_2d() -> BoxedStrategy<Patch> {
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
            .prop_map(|((fmt, dil, c, ks, pad, strides), inp)| {
                let shape = fmt.shape([1, c, inp.0, inp.1]);
                Patch::new(
                    tvec!(dil.0, dil.1),
                    tvec!(ks.0, ks.1),
                    &pad,
                    tvec![strides.0, strides.1],
                    shape.hw_dims().into(),
                    shape.hw_stride(),
                )
            })
            .boxed()
    }

    fn in_zone(coords: &[usize], zone: &[Range<usize>]) -> bool {
        for a in 0..zone.len() {
            if coords[a] < zone[a].start || coords[a] >= zone[a].end {
                return false;
            }
        }
        true
    }

    proptest! {
        #[test]
        fn test_zoning(p in patch_2d()) {
            let valid_zone = &p.valid_output_zone;
            let invalid_zones = &p.invalid_output_zones;
            for coords in ndarray::indices(&*p.output_spatial_shape) {
                let inside_valid = in_zone(coords.slice(), valid_zone);
                let invalid_count = invalid_zones.iter().filter(|z| in_zone(coords.slice(), z)).count();
                unsafe {
                    prop_assert_eq!(inside_valid, p.is_valid(coords.slice()), "coords {:?}, valid_zone: {:?} inside_valid: {:?}", coords.slice(), valid_zone, inside_valid);
                }
                if inside_valid {
                    prop_assert_eq!(invalid_count, 0);
                } else {
                    prop_assert_eq!(invalid_count, 1, "coords {:?}, valid_zone: {:?} inside_valid: {:?} invalid_zones: {:?}", coords.slice(), valid_zone, inside_valid, invalid_zones);
                }
            };
        }

        #[test]
        fn test_zone_visitor(p in patch_2d()) {
            let mut output = ndarray::ArrayD::<i32>::zeros(&*p.output_spatial_shape);
            for (c, _v) in p.visit_all_2() {
                prop_assert!(output[[c.0, c.1]] == 0);
                output[[c.0, c.1]] = 1;
            }
            assert!(output.iter().all(|&x| x == 1));
        }
    }
    #[test]
    fn test_zone_visitor_1() {
        let p = Patch::new(
            tvec!(1, 1),
            tvec![2, 1],
            &PaddingSpec::SameLower,
            tvec![1, 2],
            tvec!(2, 2),
            1,
        );
        let mut output = ndarray::ArrayD::<i32>::zeros(&*p.output_spatial_shape);
        for (c, _v) in p.visit_all_2() {
            assert!(output[[c.0, c.1]] == 0);
            output[[c.0, c.1]] = 1;
        }
        assert!(output.iter().all(|&x| x == 1));
    }
}
