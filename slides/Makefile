.PHONY: pdf clean
RM = /bin/rm -rf

# Latex utils
PDFLATEX = pdflatex
OPTIONS = --shell-escape

all: pdf

pdf:
	$(PDFLATEX) $(OPTIONS) talk.tex

clean:
	$(RM) *.aux *.log *.dvi *.pdf *.txt *.gz
