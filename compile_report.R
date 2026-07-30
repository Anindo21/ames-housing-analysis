Sys.setenv(RSTUDIO_PANDOC = "C:/Program Files/RStudio/resources/app/bin/quarto/bin/tools")
setwd("C:/Users/A01411/OneDrive - Uniper IT GmbH/Documents/University documents/Statistical Learning/ames-housing-analysis")
rmarkdown::render("ude_thesis.Rmd", output_file = "ude_thesis.pdf")
