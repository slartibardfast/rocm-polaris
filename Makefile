# ROCm Polaris — build helpers
#
# Usage:
#   make hsa-rocr          # build hsa-rocr-polaris
#   make hsa-rocr-clean    # clean + rebuild hsa-rocr-polaris
#   make hsa-rocr-install  # install (requires sudo)
#   make clean             # remove all build artifacts

PKGDIRS := hsa-rocr hip-runtime-amd rocblas-gfx803 llama.cpp

# Generic pattern rules for any package directory
%-clean:
	cd $* && rm -rf src build pkg && makepkg -sf

%-install:
	cd $* && sudo pacman -U --noconfirm $*-*-x86_64.pkg.tar.zst

%:
	@if [ -d "$@" ] && [ -f "$@/PKGBUILD" ]; then \
		cd "$@" && makepkg -sf; \
	else \
		echo "Unknown target: $@"; exit 1; \
	fi

clean:
	@for d in $(PKGDIRS); do \
		echo "Cleaning $$d..."; \
		rm -rf $$d/src $$d/build $$d/pkg; \
	done

.PHONY: clean $(PKGDIRS)
