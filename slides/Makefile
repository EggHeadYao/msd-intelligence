LATEXMK ?= latexmk
OUTDIR := build
TARGET := slides

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).pdf: $(OUTDIR)/$(TARGET).pdf
	cp $(OUTDIR)/$(TARGET).pdf $(TARGET).pdf

$(OUTDIR)/$(TARGET).pdf: $(TARGET).tex $(TARGET).bib focs-slides.sty | $(OUTDIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $(TARGET).tex

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -rf $(OUTDIR)
