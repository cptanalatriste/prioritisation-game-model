library(tidyverse)

data_file <- read_csv("strategies.csv")

font_size <- 20
geom_font_size <- font_size * (1/3)

ggplot(data = data_file, aes(x = Inflation_Probability, 
                             y = Deflation_Probability, 
                             label = paste("(", Cluster_Size, "%)"))) + 
  geom_point(mapping = aes(color = Strategy, shape = Strategy), 
             size = 8) + 
  geom_text(vjust = 0, nudge_y = 0.03, size=geom_font_size) +
  scale_shape_manual(values=seq(13,19)) +
  theme(text = element_text(size = font_size))



