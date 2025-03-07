#_______________________________________________________________________________
#
# Script to plot FA correlation results of JHU atlas
#
#_______________________________________________________________________________


# libaries 
library(ggplot2)
library(ggpubr)
library(dplyr)
library(stringr)
library(RColorBrewer)

# get path to data
localpath <- getwd()
projpath <- substr(localpath, start = 0, stop = nchar(localpath)-18) 
respath <- paste0(projpath,"/results/FA/")

# load in data for all subjects
df <- read.csv(paste0(projpath,"/Data/statsmaps/merged_fa-flair_jhu.csv"))

# run correlation across subjects for all regions
names <- unique(df$structure_name)
labels <- unique(df$label_id)

df_corr = data.frame()

for (i in 1:length(labels)){
  data = df[df$label_id == i,]
  res = cor.test(data$mean_fa, data$sum_relevance, 
                method = "pearson")
  output = c(names[i], 
             labels[i],
             cor(data$mean_fa, data$sum_relevance, method= "pearson", use = "complete.obs"),
             res$p.value)
  df_corr = rbind(df_corr, output)
}

colnames(df_corr)=c("name","id","r","p")
df_corr$r <- as.numeric(df_corr$r)
df_corr$p <- as.numeric(df_corr$p)
df_corr$id <- as.numeric(df_corr$id)
df_corr$p_bonf <- as.numeric(df_corr$p) * length(names)

colnames(df_corr)=c("name","id","r","p","p_bonf")

# save correlation across subjects for all JHU regions
write.csv(df_corr, file = paste0(respath,"corr_JHU_FA.csv"))

# ------------------------------------------------------------------------------

# Visualize results for all stat. sign. (bonferroni) regions

df_corr_bp <- subset(df_corr, df_corr$p_bonf < 0.05)
df_corr_bp<-df_corr_bp[order(df_corr_bp$r),]
df_corr_bp$name <- str_replace(df_corr_bp$name, " \\s*\\([^\\)]+\\)", "")
df_corr_bp$name <- factor(df_corr_bp$name, levels = df_corr_bp$name)  # convert to factor to retain sorted order in plot.


FA_region <- ggplot(df_corr_bp, aes(name, r)) +
  theme_classic()+
  geom_point(aes(colour = r), size=3) +
  geom_segment(aes(xend= name, yend =r, colour = r, y=0, x=name),size=1.5)+
  scale_colour_gradient2()+
  theme(text=element_text(size=20)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 15))+
  xlab("WM region") 

ggsave(paste0(respath,"FA_regions_flair.svg"),
       plot = FA_region,
       device = "svg",
       width = 31.28571,
       height = 20,
       units = "cm",
       dpi = 300
)



##### same analysis for t1-based relevance value 

# load in data for all subjects
df <- read.csv(paste0(projpath,"/Data/statsmaps/merged_fa-t1_jhu.csv"))

# run correlation across subjects for all regions
names <- unique(df$structure_name)
labels <- unique(df$label_id)

df_corr = data.frame()

for (i in 1:length(labels)){
  data = df[df$label_id == i,]
  res = cor.test(data$mean_fa, data$sum_relevance, 
                 method = "pearson")
  output = c(names[i], 
             labels[i],
             cor(data$mean_fa, data$sum_relevance, method= "pearson", use = "complete.obs"),
             res$p.value)
  df_corr = rbind(df_corr, output)
}

colnames(df_corr)=c("name","id","r","p")
df_corr$r <- as.numeric(df_corr$r)
df_corr$p <- as.numeric(df_corr$p)
df_corr$id <- as.numeric(df_corr$id)
df_corr$p_bonf <- as.numeric(df_corr$p) * length(names)

colnames(df_corr)=c("name","id","r","p","p_bonf")

# save correlation across subjects for all JHU regions
write.csv(df_corr, file = paste0(respath,"corr_JHU_FA_t1.csv"))

# ------------------------------------------------------------------------------

# Visualize results for all stat. sign. (bonferroni) regions

df_corr_bp <- subset(df_corr, df_corr$p_bonf < 0.05)
df_corr_bp<-df_corr_bp[order(df_corr_bp$r),]
df_corr_bp$name <- str_replace(df_corr_bp$name, " \\s*\\([^\\)]+\\)", "")
df_corr_bp$name <- factor(df_corr_bp$name, levels = df_corr_bp$name)  # convert to factor to retain sorted order in plot.


FA_region <- ggplot(df_corr_bp, aes(name, r)) +
  theme_classic()+
  geom_point(aes(colour = r), size=3) +
  geom_segment(aes(xend= name, yend =r, colour = r, y=0, x=name),size=1.5)+
  scale_colour_gradient2()+
  theme(text=element_text(size=20)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 15))+
  xlab("WM region") 


# save as svg
ggsave(paste0(respath,"FA_regions_t1.svg"),
       plot = FA_region,
       device = "svg",
       width = 30,
       height = 25,
       units = "cm",
       dpi = 300
)


# Assuming df1 and df2 are your two dataframes and 'name' is the column with names
df_t1 <- df_corr_bp
unique_names_df1 <- unique(df_corr_bp$name)
unique_names_df2 <- unique(df_t1$name)

# Find common names
common_names <- intersect(unique_names_df1, unique_names_df2)

# Count the number of common names
num_common_names <- length(common_names)

# Print the result
num_common_names




