# lstm-shapecalc

Tiny Rust crate for LSTM parameter counts and output shapes (PyTorch
`nn.LSTM` conventions) plus the classic Conv2d output-size formula — the
same math behind the free [heytensor.com](https://heytensor.com/) PyTorch
tensor shape calculator and RuntimeError debugger.

```rust
let n = lstm_shapecalc::params(128, 64, true); // 49408
```

MIT OR Apache-2.0
