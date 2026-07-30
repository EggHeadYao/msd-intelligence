LATEXMK ?= latexmk
OUTDIR := build
TARGET := poster
TEX_SOURCES := $(TARGET).tex
STYLE_SOURCES := focs-poster.sty
IMAGE_SOURCES := $(wildcard img/*)

.PHONY: all clean distclean

all: $(TARGET).pdf

$(TARGET).pdf: $(OUTDIR)/$(TARGET).pdf
	cp $(OUTDIR)/$(TARGET).pdf $(TARGET).pdf

$(OUTDIR)/$(TARGET).pdf: $(TEX_SOURCES) $(STYLE_SOURCES) $(IMAGE_SOURCES) | $(OUTDIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(OUTDIR) $(TARGET).tex

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -rf $(OUTDIR)

distclean: clean
	rm -f $(TARGET).pdf
