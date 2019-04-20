use crate::internal::*;

#[derive(Debug, Clone)]
pub enum PaddingSpec {
    Explicit(TVec<usize>, TVec<usize>),
    Valid,
    SameUpper,
    SameLower,
}

impl Default for PaddingSpec {
    fn default() -> PaddingSpec {
        PaddingSpec::Valid
    }
}

#[derive(Debug, Clone, new, PartialEq)]
pub struct ComputedPaddedDim<D: DimLike> {
    pub pad_before: D,
    pub pad_after: D,
    pub output: D,
}

#[derive(Debug, Clone)]
pub struct ComputedPaddedDims<D: DimLike> {
    pub dims: TVec<ComputedPaddedDim<D>>,
}

impl PaddingSpec {
    pub fn valid_dim(&self, d: usize) -> bool {
        match self {
            PaddingSpec::Valid => true,
            PaddingSpec::Explicit(a, b) => a[d] == 0 && b[d] == 0,
            _ => false,
        }
    }

    pub fn rm_axis(&self, d: usize) -> PaddingSpec {
        match self {
            PaddingSpec::Explicit(a, b) => {
                let mut a = a.clone();
                let mut b = b.clone();
                a.remove(d);
                b.remove(d);
                PaddingSpec::Explicit(a, b)
            }
            _ => self.clone(),
        }
    }

    pub fn compute<D: DimLike>(
        &self,
        input_spatial_shape: &[D],
        kernel_spatial_shape: &[usize],
        dilations: &[usize],
        strides: &[usize],
    ) -> ComputedPaddedDims<D> {
        let dims = (0..input_spatial_shape.len())
            .map(|d| {
                self.compute_one(
                    d,
                    input_spatial_shape[d],
                    kernel_spatial_shape[d],
                    dilations[d],
                    strides[d],
                )
            })
            .collect();
        ComputedPaddedDims { dims }
    }

    pub fn compute_one<D: DimLike>(
        &self,
        axis: usize,
        input_spatial_dim: D,
        kernel_spatial_dim: usize,
        dilation: usize,
        stride: usize,
    ) -> ComputedPaddedDim<D> {
        match self {
            PaddingSpec::Valid => {
                Self::explicit(input_spatial_dim, kernel_spatial_dim, dilation, stride, 0, 0)
            }
            PaddingSpec::Explicit(ref bef, ref aft) => Self::explicit(
                input_spatial_dim,
                kernel_spatial_dim,
                dilation,
                stride,
                bef[axis],
                aft[axis],
            ),
            PaddingSpec::SameUpper => {
                Self::same(input_spatial_dim, kernel_spatial_dim, dilation, stride, true)
            }
            PaddingSpec::SameLower => {
                Self::same(input_spatial_dim, kernel_spatial_dim, dilation, stride, false)
            }
        }
    }

    fn explicit<D: DimLike>(
        data_spatial_dim: D,
        kernel_spatial_dim: usize,
        dilation: usize,
        stride: usize,
        bef: usize,
        aft: usize,
    ) -> ComputedPaddedDim<D> {
        let kernel_field = (kernel_spatial_dim - 1) * dilation + 1;
        let dim = (data_spatial_dim + bef + aft - kernel_field + 1).div_ceil(stride);
        ComputedPaddedDim::new(dim, bef.into(), aft.into())
    }

    fn same<D: DimLike>(
        data_spatial_dim: D,
        kernel_spatial_dim: usize,
        dilation: usize,
        stride: usize,
        upper: bool,
    ) -> ComputedPaddedDim<D> {
        let dim = data_spatial_dim.div_ceil(stride);
        let kernel_field = (kernel_spatial_dim - 1) * dilation + 1;
        let pad = if stride <= kernel_field {
            (dim - 1) * stride + kernel_field - data_spatial_dim
        } else {
            D::zero()
        };
        let lower_pad = pad / 2;
        let higher_pad = pad - pad / 2;
        let (before, after) = if upper {
            (lower_pad, higher_pad)
        } else {
            (higher_pad, lower_pad)
        };
        ComputedPaddedDim::new(before, after, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_stride_1() {
        assert_eq!(PaddingSpec::same(1usize, 2usize, 1, 1, true), ComputedPaddedDim::new(1, 0, 1));
        assert_eq!(PaddingSpec::same(2usize, 2usize, 1, 1, true), ComputedPaddedDim::new(2, 0, 1));
        assert_eq!(PaddingSpec::same(3usize, 2usize, 1, 1, true), ComputedPaddedDim::new(3, 0, 1));
        assert_eq!(PaddingSpec::same(4usize, 2usize, 1, 1, true), ComputedPaddedDim::new(4, 0, 1));
    }

    #[test]
    fn same_stride_2() {
        assert_eq!(PaddingSpec::same(1usize, 2usize, 1, 2, true), ComputedPaddedDim::new(1, 0, 1));
        assert_eq!(PaddingSpec::same(2usize, 2usize, 1, 2, true), ComputedPaddedDim::new(1, 0, 0));
        assert_eq!(PaddingSpec::same(3usize, 2usize, 1, 2, true), ComputedPaddedDim::new(2, 0, 1));
        assert_eq!(PaddingSpec::same(4usize, 2usize, 1, 2, true), ComputedPaddedDim::new(2, 0, 0));
    }

    #[test]
    fn same_1() {
        assert_eq!(PaddingSpec::same(6usize, 1usize, 1, 2, true), ComputedPaddedDim::new(3, 0, 0));
    }

    #[test]
    fn same_ker_3() {
        assert_eq!(PaddingSpec::same(1usize, 3usize, 1, 1, true), ComputedPaddedDim::new(1, 1, 1));
        assert_eq!(PaddingSpec::same(2usize, 3usize, 1, 1, true), ComputedPaddedDim::new(2, 1, 1));
        assert_eq!(PaddingSpec::same(3usize, 3usize, 1, 1, true), ComputedPaddedDim::new(3, 1, 1));
        assert_eq!(PaddingSpec::same(4usize, 3usize, 1, 1, true), ComputedPaddedDim::new(4, 1, 1));
    }
}
