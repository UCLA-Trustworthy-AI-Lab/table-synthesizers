# Table Synthesis Presentation

This directory contains a Beamer presentation on advanced table synthesis for Amazon data, focusing on diffusion and LLM-based models.

## Files

- `presentation.tex` - Main LaTeX source file for the presentation
- `Makefile` - Build automation for compiling the presentation
- `highlevel_outline` - High-level outline and structure notes
- `why diffusion beats trees` - Technical comparison document
- `expalnation of synthesizer repo` - Details about the synthesizer implementations

## Compilation

### Prerequisites

Install a LaTeX distribution with Beamer support:

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
# Or for a lighter installation:
sudo apt-get install texlive-latex-extra texlive-fonts-recommended texlive-latex-recommended
```

**macOS:**
```bash
brew install --cask mactex
# Or for a lighter installation:
brew install --cask basictex
```

### Building the Presentation

Using make:
```bash
make
```

Or directly with pdflatex:
```bash
pdflatex presentation.tex
pdflatex presentation.tex  # Run twice for proper references
```

### Cleaning Build Files

```bash
make clean      # Remove auxiliary files
make distclean  # Remove all generated files including PDF
```

## Presentation Overview

**Title:** Advanced Table Synthesis for Amazon Data

**Duration:** 30 minutes (30 slides)

**Target Audience:** Amazon engineers participating in internal hackathon on table reasoning

**Key Message:** Amazon needs table synthesis, and the best choice is diffusion/LLM-based models for complex, high-dimensional, text-rich data.

## Content Structure

1. **Opening (3-4 slides):** Team introduction and agenda
2. **Problem Statement (3-4 slides):** Data scarcity and privacy challenges at Amazon
3. **Solution Overview:** Table synthesis technology
4. **Technical Analysis:** Why diffusion/LLM models outperform tree-based methods
5. **Experiments:** Benchmarks on Amazon tables with metrics for fidelity, utility, and privacy
6. **Case Studies:** Guidelines for small (<1K), medium (1K-50K), and large (>50K) tables
7. **Implementation:** How to use the table synthesizer library
8. **Q&A**

## Key Technical Points

- **Tree-based methods:** Good for simple, single tables with low cardinality
- **Diffusion models:** Best for complex correlations and high-cardinality data
- **LLM-based models:** Excel with text data and multi-table relationships
- **Privacy guarantees:** Differential privacy integration available

## Using the Table Synthesizer Library

The presentation includes code examples for using the library:

```python
from stg import TableSynthesizer

# Simple usage
synthesizer = TableSynthesizer('TabDDPM')
synthesizer.fit(df, epochs=100)
synthetic_df = synthesizer.sample(n=10000, return_dataframe=True)
```

For more details, see the library repository at `/home/xiaofeng/table_reasoning/table-synthesizers/`