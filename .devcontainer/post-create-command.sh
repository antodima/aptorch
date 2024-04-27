#!/bin/sh
chmod -R +x ./.vscode
poe install_dev_deps
poe install_precommit
poe start_shell
