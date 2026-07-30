LATEXMK ?= latexmk
OUTDIR := build
TARGET := slides
TEX_SOURCES := $(TARGET).tex artistdistance_slides.tex
BIB_SOURCES := $(TARGET).bib model_refs.bib model_refs_story.bib
STYLE_SOURCES := focs-slides.sty

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).pdf: $(OUTDIR)/$(TARGET).pdf
	cp $(OUTDIR)/$(TARGET).pdf $(TARGET).pdf

$(OUTDIR)/$(TARGET).pdf: $(TEX_SOURCES) $(BIB_SOURCES) $(STYLE_SOURCES) | $(OUTDIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $(TARGET).tex

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -rf $(OUTDIR)
