# Release process

1. Ensure the working tree contains only intended release changes.
2. Update the workspace version in `Cargo.toml` and the matching section in `CHANGELOG.md`.
3. Run the full local verification:

   ```powershell
   .\scripts\cargo.ps1 fmt --all -- --check
   .\scripts\cargo.ps1 clippy --workspace --all-targets '--' -D warnings
   .\scripts\cargo.ps1 test --workspace --locked
   .\scripts\package-release.ps1 -Version 1.0.0
   ```

4. Test the ZIP on a clean Windows x64 machine with current CPU and GPU drivers.
5. Commit and push the release changes. Wait for the `CI` workflow to pass.
6. For a manual release, use the tested ZIP and SHA-256 checksum in `dist/`. Create the GitHub release against the verified source commit and attach both files.

Pushing a `v*.*.*` tag starts the automatic `Release` workflow, which builds, packages, and publishes the assets. Coordinate the tag and manual upload so both paths target the same release.

Confirm that the release archive contains `LICENSE` and that dependency licensing remains compatible with distribution.
