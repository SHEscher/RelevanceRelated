#_______________________________________________________________________________
#
# Script to plot subcortical GMV correlation results
#
#_______________________________________________________________________________


library(ggseg)
library(cowplot)
library(tidyverse)

# get path to data
localpath <- getwd()
projpath <- substr(localpath, start = 0, stop = nchar(localpath)-18) 
respath <- paste0(projpath,"/results/GM/")

df <- read.csv(paste0(respath,"corr_subcort_vol-relevance.csv"))
colnames(df) = c("label","r","p")
# label issue
df$label[7] <-"x3rd-ventricle"
df$p_bonf <- as.numeric(df$p) * nrow(df)


# plot subcortical GMV correlation on aseg segmentation using ggseg
p_sub_gm<-ggplot(df) +
  theme_void() +
  geom_brain(atlas = aseg, 
             #position = position_brain(hemi ~ side),
             aes(fill = r)) +
  scale_fill_gradient2() +
  theme(text = element_text(size = 15)) +
  theme(plot.title = element_text(size = rel(1.5),face='bold'))
  

# load in edited data (labels are edited for visualization purpose)

df1 <- read.csv(paste0(respath,"corr_subcort_vol-relevance_plt.csv"))
df1$p_bonf <- as.numeric(df1$p) * nrow(df1)
df1 <- subset(df1, p_bonf<0.05)
df1<-df1[order(df1$r),]

df1$label_name <- factor(df1$label_name, levels = df1$label_name)  


p_sub_gm2 <- ggplot(df1, aes(label_name, r)) +
  theme_classic()+
  geom_point(aes(colour = r), size=3) +
  geom_segment(aes(xend= label_name, yend =r, colour = r, y=0, x=label_name),size=1.5)+
  scale_colour_gradient2()+
  theme(text=element_text(size=20)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 15))+
  xlab("region") +
  theme(panel.grid.major = element_line(color = "grey",
                                        size = 0.3,
                                        linetype = 2))+
  theme(legend.position = "none")


fig = ggdraw() +
  draw_plot(p_sub_gm, x = 0, y = 0.6, width = 1, height = 0.4) +
  draw_plot(p_sub_gm2, x = 0, y = 0, width = 1, height = 0.6)

# save as png
ggsave(paste0(respath,"figure4.png"),
       plot = fig,
       device = "png",
       width = 32,
       height = 25,
       units = "cm",
       dpi = 500
)

# save as svg
ggsave(paste0(respath,"figure4.svg"),
       plot = fig,
       device = "svg",
       width = 32,
       height = 25,
       units = "cm",
       dpi = 500
)

# for results report
# mean of pos/neg r and range
df_pos = subset(df, r>0)
df_neg = subset(df, r<0)

print(mean(df_pos$r))
print(mean(df_neg$r))
print(min(df_pos$r))
print(max(df_pos$r))
print(min(df_neg$r))
print(max(df_neg$r))
