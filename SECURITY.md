# Security

## Reporting a problem

Please report security problems privately through the repository's **Security → Report a
vulnerability** page on GitHub, not in a public issue. Expect a first reply within a week.

## What this project does and doesn't touch

- The library is pure computation on arrays you pass in. It opens no files, makes no network
  requests and runs no code from its inputs.
- The website's calculator runs entirely in the browser. Trade results entered there are
  not uploaded or stored.
- Releases are published to PyPI through trusted publishing from GitHub Actions. No tokens
  or passwords are stored in the repository.

## Supported versions

Only the latest release receives fixes.
