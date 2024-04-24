#!/bin/sh

huggingface-cli download TheBloke/zephyr-7B-beta-GGUF zephyr-7b-beta.Q5_K_M.gguf --local-dir models/ --local-dir-use-symlinks False
