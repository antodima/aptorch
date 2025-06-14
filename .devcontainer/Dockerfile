ARG VARIANT="3.12-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

ARG INSTALL_DEV_TOOLS
ARG POETRY_VERSION
ARG POETRY_CORE_VERSION
ARG POETHEPOET_VERSION
ARG PRE_COMMIT_VERSION

RUN if [ "${INSTALL_GIT}" != "no" ]; \
    then \
        apt-get update && \
        apt-get install --no-install-recommends -y git && \
        git config --global --add safe.directory '*'; \
    fi

RUN if [ "${INSTALL_DEV_TOOLS}" != "no" ]; \
    then \
        pip install poetry==${POETRY_VERSION} poetry-core==${POETRY_CORE_VERSION} poethepoet==${POETHEPOET_VERSION} pre-commit==${PRE_COMMIT_VERSION}; \
    fi
