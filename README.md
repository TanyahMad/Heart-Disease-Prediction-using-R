# Install required R packages
R -e "install.packages(c('shiny','tidyverse','caret','randomForest','corrplot','ggplot2','e1071','gbm'))"

# Run the Shiny app
R -e "shiny::runApp('.')"
