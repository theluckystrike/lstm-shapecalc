//! LSTM shape math matching PyTorch `nn.LSTM` conventions.
//! Useful when debugging `RuntimeError: size mismatch`.

/// Trainable parameter count of an LSTM layer.
/// PyTorch: 4 gates, each with `hidden*(input+hidden)` weights, plus
/// 4 biases of `hidden` when `bias` is true.
///
/// ```
/// assert_eq!(lstm_shapecalc::params(10, 20, true), 2480);
/// ```
pub fn params(input: usize, hidden: usize, bias: bool) -> usize {
    let p = 4 * hidden * (input + hidden);
    if bias { p + 4 * hidden } else { p }
}

/// Output shape of an LSTM layer: `(seq_len, hidden)` — independent of input size.
///
/// ```
/// assert_eq!(lstm_shapecalc::output_shape(50, 64), (50, 64));
/// ```
pub fn output_shape(seq_len: usize, hidden: usize) -> (usize, usize) {
    (seq_len, hidden)
}

/// Conv2d output spatial size: floor((in + 2*pad - kernel)/stride) + 1.
///
/// ```
/// assert_eq!(lstm_shapecalc::conv2d_out(112, 3, 2, 1), 56);
/// ```
pub fn conv2d_out(input: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (input + 2 * pad - kernel) / stride + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lstm_params() { assert_eq!(params(10, 20, true), 2480); assert_eq!(params(10, 20, false), 2400); }
    #[test]
    fn lstm_out() { assert_eq!(output_shape(50, 64), (50, 64)); }
    #[test]
    fn conv() { assert_eq!(conv2d_out(112, 3, 2, 1), 56); assert_eq!(conv2d_out(224, 7, 2, 3), 112); }
}
