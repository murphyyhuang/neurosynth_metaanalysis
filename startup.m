addpath /home/libilab2/a/users/huan1282/dev/src/libilab/hgwen
addpath /home/libilab2/a/users/huan1282/dev/src/libilab/cifti
addpath /home/libilab/a/software/workbench
addpath /home/libilab/a/matlab/0libi/TomLu
addpath (genpath('/home/libilab/a/matlab/0libi/zmliu'));

addpath /home/libilab/a/software/spm12
addpath(fullfile(pwd, './BrainGraphTools'));
addpath(fullfile(pwd, './cnm'));

fsldir = getenv('FSLDIR');
fsldirmpath = sprintf('%s/etc/matlab',fsldir);
path(path, fsldirmpath);
