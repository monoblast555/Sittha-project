#Protein
#read csv files
prot1 <- read.csv("prot_bio1.csv", header = TRUE, check.names = F)
prot2 <- read.csv("prot_bio2.csv", header = TRUE, check.names = F)
prot3 <- read.csv("prot_bio3.csv", header = TRUE, check.names = F)
prot4 <- read.csv("prot_bio4.csv", header = TRUE, check.names = F)

#Join all protein files to one file
library(tidyverse)
joined_prot <- full_join(prot1, prot2, by = c("Samples"))
joined_prot <- full_join(joined_prot, prot3, by = c("Samples"))
joined_prot <- full_join(joined_prot, prot4, by = c("Samples"))
#Write to csv
write.csv(joined_prot, "D:\\Research project\\pool_data\\prot.csv", row.names = F)

#Find first rows with data
first_prot <- which(apply(joined_prot[, -1], 1, function(row) any(row != 0)))[1]
joined_prot[first_prot, ]

#Find last rows with data
last_prot <- which(apply(joined_prot[, -1], 1, function(row) any(row != 0)))[length(which(apply(joined_prot[, -1], 1, function(row) any(row != 0))))]
joined_prot[last_prot, ]



#Lipid
#read csv files
lip1 <- read.csv("lipid_bio1.csv", header = TRUE, check.names = F)
lip2 <- read.csv("lipid_bio2.csv", header = TRUE, check.names = F)
lip3 <- read.csv("lipid_bio3.csv", header = TRUE, check.names = F)
lip4 <- read.csv("lipid_bio4.csv", header = TRUE, check.names = F)

joined_lip <- full_join(lip1, lip2, by = c("Samples"))
joined_lip <- full_join(joined_lip, lip3, by = c("Samples"))
joined_lip <- full_join(joined_lip, lip4, by = c("Samples"))

#Find first rows with data
first_lip <- which(apply(joined_lip[, -1], 1, function(row) any(row != 0)))[1]
joined_lip[first_lip, ]


#Find last rows with data
last_lip <- which(apply(joined_lip[, -1], 1, function(row) any(row != 0)))[length(which(apply(joined_lip[, -1], 1, function(row) any(row != 0))))]
joined_lip[last_lip, ]




#trim data 
trimmed_prot <- joined_prot[first_prot:last_prot, ]

trimmed_prot <- trimmed_prot %>% drop_na()
write.csv(trimmed_prot,"trimmed_prot_no_na.csv", row.names = F)

#lipid data have problems, do it later

#baseline corrected and plot by Matlab

library(tidyverse)
library(dplyr)
library(readr)
library(tidyr)


# Normalisation
# Load corrected spectra
corrected <- read.csv("corrected_prot.csv", header = T, check.names = F)

#Convert to long format
long_data <- corrected %>%
  pivot_longer(cols = -mz, names_to = "idx", values_to = "Intensity") %>%
  mutate(idx = as.integer(idx))  # Convert for merging

#Normalize by sqr sum of square
normalized_data <- long_data %>%
  group_by(idx) %>%
  mutate(Norm_Intensity = Intensity / sqrt(sum(Intensity^2))) %>%
  ungroup()

# read metadata
metadata <- read.csv("metadata.csv", header = T)

# merge metadata with nomalised data
merged_data <- normalized_data %>%
  left_join(metadata, by = "idx")

# save file
write.csv(merged_data, "normal_prot_w_meta.csv", row.names = F)

#try to plot
ggplot(merged_data %>% filter(idx == 1), aes(x = mz, y = Norm_Intensity)) +
  geom_line() +
  labs(title = "Normalized Spectrum for Sample 1") +
  theme_minimal()

#rename sample in new column
merged_data$sample_id <- paste(merged_data$strains, merged_data$bio_rep, merged_data$idx, sep = "-")

# create new df to select only m/z, intensities, id
iden <- merged_data
iden <- data.frame(iden$mz, iden$Norm_Intensity, iden$sample_id)
colnames(iden) <- c("mz", "norm_intens", "sample_id")

#flip m/z back to column to export to a new file, plot with Matlab later
wide_data <- iden %>%
  select(mz, sample_id, norm_intens) %>%
  pivot_wider(names_from = sample_id, values_from = norm_intens)

#save file
write.csv(wide_data, "norm_prot.csv", row.names = F)
