#!/usr/bin/env bash

set -eux

cd $(dirname $0)

rustup run $(NIGHTLY_TOOLCHAIN) cargo doc

# cargo-deadlinks will check any links in docs generated by `cargo doc`.
# This is useful as rustdoc uses raw links which are error prone.
command -v cargo-deadlinks &> /dev/null &&
	cargo deadlinks

cd -
