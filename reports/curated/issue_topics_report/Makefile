TEX=report.tex
PDF=report.pdf
GREENE_REPORT_DIR ?= /vast/sxr203/newspaper-downloads/dedupe-webp/issue_topics_report_latex_mechanics_v2_gpt5nano_legend_outside_20260130_002431

.PHONY: all clean sync

all: $(PDF)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

sync:
	rsync -av --delete --exclude README.md --exclude Makefile --exclude .gitignore greene:$(GREENE_REPORT_DIR)/ ./

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot $(PDF)
