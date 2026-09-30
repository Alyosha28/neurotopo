# Git metadata sidecars on this mounted workspace

2026-09-24. While staging EXP-0009, Git emitted `non-monotonic index` for `.git/objects/pack/._pack-…idx`. `file` identified that file and three neighboring `._` files as AppleDouble metadata; the normal `pack-…idx/.pack/.rev` and `multi-pack-index` remained present. Four sidecars were moved, byte-preserving, to `/private/tmp/neurotopo-git-appledouble-quarantine-20260924/`.

`git fsck --full --no-reflogs` then found eight invalid `refs/**/._*` entries. A read-only audit found 35 remaining regular `._*` files under `.git`, all with AppleDouble magic `00 05 16 07`. They were moved with their relative paths to `/private/tmp/neurotopo-git-appledouble-quarantine-20260924/remaining_git/`. No normal Git object, ref, index or pack was moved or deleted. After isolation, `git fsck --full --no-reflogs` exited0 with only three dangling, unreachable objects; staged diff and status worked without sidecar errors.

The quarantine is recoverable and local to this machine. Do not infer research-data duplication from these metadata files or broaden this cleanup to non-AppleDouble Git files. Formal EXP-0009 execution still uses a separate clean committed checkout; this repair only restores reliable local Git bookkeeping.
