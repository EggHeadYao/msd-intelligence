LATEXMK ?= latexmk
OUTPUT_ROOT ?= ../../docs_output
OUTDIR ?= $(OUTPUT_ROOT)/build
TARGET := slides
OUTPUT_PDF := $(OUTPUT_ROOT)/$(TARGET).pdf
C2_FIGURES := img/c2/c2_graph_schema.pdf \
	img/c2/c2_walk_policy.pdf \
	img/c2/c2_masked_results.pdf \
	img/c2/c2_figures_metadata.json

.PHONY: all clean

all: $(OUTPUT_PDF)

$(OUTPUT_PDF): $(OUTDIR)/$(TARGET).pdf | $(OUTPUT_ROOT)
	cp $(OUTDIR)/$(TARGET).pdf $(OUTPUT_PDF)

$(OUTDIR)/$(TARGET).pdf: $(TARGET).tex $(TARGET).bib focs-slides.sty $(C2_FIGURES) | $(OUTDIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $(TARGET).tex

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(OUTPUT_ROOT):
	mkdir -p $(OUTPUT_ROOT)

clean:
	rm -rf $(OUTDIR) $(OUTPUT_PDF)
