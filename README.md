[![crates.io link](https://img.shields.io/crates/v/wasmi.svg)](https://crates.io/crates/wasmi)
[![Build Status](https://travis-ci.org/paritytech/wasmi.svg?branch=master)](https://travis-ci.org/paritytech/wasmi)

# `wasmi`

`wasmi` - a Wasm interpreter.

`wasmi` was conceived as a component of [parity-ethereum](https://github.com/paritytech/parity-ethereum) (ethereum-like contracts in wasm) and [substrate](https://github.com/paritytech/substrate). These projects are related to blockchain and require a high degree of correctness, even if that might be over conservative. This specifically means that we are not trying to be involved in any implementation of any of work-in-progress Wasm proposals. We are also trying to be as close as possible to the spec, which means we are trying to avoid features that is not directly supported by the spec. This means that it is flexible on the one hand and on the other hand there shouldn't be a problem migrating to another spec compliant execution engine.

With all that said, `wasmi` should be a good option for initial prototyping.

# Build & Test

As `wasmi` contains a git submodule, you need to use `--recursive` for cloning or to checkout the submodule explicitly, otherwise the testing would fail.

```
git clone https://github.com/paritytech/wasmi.git --recursive
cd wasmi
cargo build
cargo test
```

# `no_std` support

This crate supports `no_std` environments.
Enable the `core` feature and disable default features:
```toml
[dependencies.wasmi]
version = "*"
default-features = false
features = ["core"]
```

When the `core` feature is enabled, code related to `std::error` is disabled.

# License

`wasmi` is primarily distributed under the terms of both the MIT
license and the Apache License (Version 2.0), at your choice.

See LICENSE-APACHE, and LICENSE-MIT for details.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `wasmi` by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
