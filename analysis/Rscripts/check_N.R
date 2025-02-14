# ------------------------------------------------------------------------------
# Check availability of data and data quality
# Author: Frauke Beyer
# ------------------------------------------------------------------------------

library(tidyverse)

setwd(".../RelevanceRelated/")  # anonymized
# get path to data
localpath <- getwd()
respath <- paste0(localpath,"/results/PVS/")

dkt=read.csv("./data/statsmaps/merged_cs_dktbase.csv")
dkt40=read.csv("./data/statsmaps/merged_cs_dkt.csv")
subcort=read.csv("./data/statsmaps/merged_subcort_aseg.csv")
fa=read.csv("./data/statsmaps/merged_fa_jhu.csv")
df_original_flair_comb <- read.csv(paste0(respath,"T1-FLAIR.PVS_flair-subens.csv"))
df_original_t1_comb <- read.csv(paste0(respath,"T1-FLAIR.PVS_t1-subens.csv"))

#Simons original subjects
orig=read.csv(".../Data/subject_tables/sic_tab_reduced.csv")  # anonymized
summary=read.csv("./data/subject_tables/sic_available_mri_tab_relevance_related.csv")

#SIC to Pseudo-ID transform
pseudo=read.csv(".../Data/pseudo_mrt_20201214.csv")  # anonymized

#QUALITY & OVERVIEW
ov=read.csv(".../Preprocessed/derivatives/..._subject_list_inclusion_exclusion....csv") # anonymized
#FreeSurfer:
fs=read.csv(".../Preprocessed/derivatives/FreeSurfer/.../freesurfer_qa_....csv")  # anonymized
#DWI:
dwi=read.csv(".../Lipsia_preprocessing/qa/Subject_list_DTI_comments_2018_....csv") # anonymized

###################APARC########################################################
##MERGE:
t=merge(orig, pseudo, by.x="SIC_FS", by.y="sic", all.x=T)

f=merge(t, fs, all.x=T)
table(f$BL_usable)#N=1992 should be usable with FS

#Check why only so few have dkt (because less have aparcDKTatlas40, don't know why)
red=dkt[dkt$hemisphere=="left"&dkt$structure_name=="cuneus",]
#red40=dkt40[dkt$hemisphere=="left"&dkt40$structure_name=="cuneus",]

f_dkt=merge(f, red, by.x="pseudonym", by.y="subject", all.x=T)
#f_dkt40=merge(f, red40, by.x="pseudonym", by.y="subject", all.x=T)

nrow(f_dkt[!is.na(f_dkt$average_thickness_mm),])
nrow(f_dkt[is.na(f_dkt$average_thickness_mm),])#498 are missing who should be available!!
f_dkt$missing_dkt=as.numeric(is.na(f_dkt$average_thickness_mm))
#update: when using dktbase, only 137 are missing -> MISSING RELEVANCE MAP

# dktatlas40=c()
# notdktatlas40=c()
# j=1
# i=1
# for (subj in f_dkt[is.na(f_dkt$average_thickness_mm),"pseudonym"]){
#   if (file.exists(paste0(".../freesurfer_all/",subj,"/stats/lh.aparc.DKTatlas40.stats"))){  # anonymized
#     print(paste0(subj," exists"))
#     dktatlas40[i]=subj
#     i=i+1
#   }else{print(paste0(subj," doesn't exist"))
#    notdktatlas40[j]=subj
#   j=j+1}
# }
#
# for (subj in f_dkt[is.na(f_dkt$average_thickness_mm),"pseudonym"]){
#   if (file.exists(paste0(".../freesurfer_all/",subj,"/stats/lh.aparc.stats"))){  # anonymized
#     print(paste0(subj," exists"))
#   }else{print(paste0(subj," doesn't exist"))}
# }
# files=list.files(path=, recursive = T)
#
# #Collect all missing subjects in summary table
orig_miss=merge(t, f_dkt[,c("SIC_FS", "missing_dkt")], all.x=T)

########################ASEG##############################-> all are available, but should exclude those for
#whom FREESURFER is not usable.
f_aseg=merge(f, subcort[subcort$label_name=="Right-Lateral-Ventricle",], by.x="pseudonym", by.y="subject", all.x=T)
f_aseg=f_aseg[!duplicated(f_aseg$pseudonym),]
f_aseg$missing_aseg_rel=as.numeric(is.na(f_aseg$mean_relevance))
f_aseg$missing_aseg=as.numeric(is.na(f_aseg$source_basename))
orig_miss=merge(orig_miss, f_aseg[,c("SIC_FS", "missing_aseg")], all.x=T)

####FA####

f=merge(t, dwi, all.x=T, by.x="SIC_FS", by.y="SIC")
tmp=fa[fa$structure_name=="Genu of corpus callosum",]
tmp=tmp[!duplicated(tmp$subject),]
f=merge(f,tmp, all.x=T, by.x="SIC_FS", by.y="subject")
nrow(f[is.na(f$mean_relevance),])#242 are missing from N=2016
nrow(f[is.na(f$mean_fa),])

table(f[is.na(f$mean_relevance),"DWI_useful"])#97 DWI is not completed
table(f[is.na(f$mean_relevance)&f$DWI_useful_y_n=="Yes","DWI_comments_Lipsia_pipeline"])#132 are ok
f$missing_fa=as.numeric(is.na(f$mean_relevance)&f$DWI_useful_y_n=="Yes")
nrow(f[is.na(f$mean_relevance)&f$DWI_useful_y_n=="Yes"&f$DWI_comments_Lipsia_pipeline=="ok", ])
orig_miss=merge(orig_miss, f[,c("SIC_FS", "missing_fa", "DWI_comments_Lipsia_pipeline")], all.x=T)

###################PVS####################################
f=merge(t, df_original_flair_comb,  all.x=T, by.x="SIC_FS", by.y="SIC")
nrow(f[(is.na(f$p_pvs_voxel)),]) #144 are missing PVS values, but PVS segmentations are there. Why?
f$missing_pvs=as.numeric(is.na(f$p_pvs_voxel))
orig_miss=merge(orig_miss, f[,c("SIC_FS", "missing_pvs")], all.x=T)

f=merge(t, df_original_t1_comb,  all.x=T, by.x="SIC_FS", by.y="SIC")
nrow(f[(is.na(f$p_pvs_voxel)),])


## MISSING relevance for all
#nrow(orig_miss[orig_miss$missing_dkt==1&orig_miss$missing_fa==1&
#                 orig_miss$missing_pvs==1&orig_miss$missing_aseg==1,])

#write.csv(orig_miss[orig_miss$missing_dkt==0|orig_miss$missing_aseg==0,"SIC_FS"], "./analysis/tests/missing_aseg_dkt.csv")
#write.csv(orig_miss[orig_miss$missing_pvs==0|orig_miss$missing_fa==0,"SIC_FS"], "./analysis/tests/missing_pvs_fa.csv")

#nrow(orig_miss[orig_miss$missing_dkt==1&orig_miss$missing_fa==0&
#            orig_miss$missing_pvs==0&orig_miss$missing_aseg==1,])


rr=read.csv("./data/subject_tables/relevance_related_data_overview.csv")

##CHECK missing
#17 with DWI ok, FLAIR relevance map ok but missing_fa.
table(rr[rr$missing_fa==1&rr$missing_flair_relevance_map==0,"DWI_comments_Lipsia_pipeline"])


for (subj in rr[rr$missing_fa==1&rr$missing_flair_relevance_map==0,"SIC_FS"]){
  if (file.exists(paste0(".../Lipsia_preprocessing/mri/", subj, "/", subj,"_fa.nii.gz"))){  # anonymized
    print(paste0(subj," exists"))
  }else{print(paste0(subj," doesn't exist"))}
}

#MODELS for age & sex differences
#Age
testage_t1=lm(AGE_FS~missing_t1_relevance_map, data=rr)
testage_flair=lm(AGE_FS~missing_flair_relevance_map, data=rr)

#Sex
table(rr$sex,rr$missing_t1_relevance_map)
dat=rbind(c(62,884),c(75,994))
dimnames(dat)<-list(gender=c("F", "M"), missing=c("yes", "no"))
chisq.test(dat)

table(rr$sex,rr$missing_flair_relevance_map)
dat=rbind(c(64,882),c(76,993))
dimnames(dat)<-list(gender=c("F", "M"), missing=c("yes", "no"))
chisq.test(dat)
