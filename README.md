# pytorch-ci

Out-of-tree CI for PyTorch on RISC-V.

## Overview

This repository is a downstream CI for [pytorch/pytorch](https://github.com/pytorch/pytorch), registered as a [PyTorch Out-of-Tree CI](https://github.com/pytorch/pytorch/blob/main/.github/allowlist.yml#L40).

The relay mechanism is defined in [RFC #90](https://github.com/pytorch/rfcs/pull/90), which describes the Cross-Repo CI Relay (CRCR) architecture: GitHub webhooks from `pytorch/pytorch` are forwarded via `repository_dispatch` events to registered downstream repositories at increasing trust levels (L1–L4).

## Workflows

### `out-of-tree-ci.yml` — Dispatch receiver

Listens for `repository_dispatch` (push) events forwarded by the CRCR relay from `pytorch/pytorch`. Triggers the main `ci.yml` workflow when any of the following conditions match:

- A commit lands on `refs/heads/main` (via `refs/tags/trunk/` tag)
- The `ciflow/docker` label is applied (Docker image changes in `.ci/docker`)
- The `ciflow/riscv64` label is applied

### `ci.yml` — Build pipeline

Triggered by the dispatch receiver (or manually via `workflow_dispatch`) with a `ref`/`sha` pair pointing to a commit in `pytorch/pytorch`.

**`docker-builds` job** — Runs on `ubuntu-24.04`. Builds and pushes a cross-compilation Docker image (`pytorch-linux-noble-riscv64-py3.12-gcc14`) to GHCR, using QEMU + Docker Buildx. Skips the build if the image for the current `.ci/docker` tree hash already exists.

**`linux-noble-riscv64-py3_12-build` job** — Runs on `ubuntu-24.04-riscv` self-hosted runners. Checks out `pytorch/pytorch` at the given ref, merges [pytorch/pytorch#182278](https://github.com/pytorch/pytorch/pull/182278) (in-flight RISC-V patches), then builds PyTorch inside the Docker container using `.ci/pytorch/build.sh`. Build artifacts (`dist/`, `build/lib`, `build/bin`) are uploaded as GitHub Actions artifacts with a 14-day retention.

## Build caching

The build uses a custom [sccache](https://github.com/luhenry/sccache) binary with a Redis-backed distributed coordinator, allowing multiple concurrent RISC-V build jobs to share compilation work. The cache bucket is hosted on Scaleway Object Storage (`s3.fr-par.scw.cloud`).
