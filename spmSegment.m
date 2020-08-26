function out    = spmSegment(structural_fn, spm_dir, dcm2nii_dir)
    %Based on https://github.com/jsheunis/matlab-spm-scripts-jsh
    %Calls SPM_jobman to segment the wm & gm of the supplied structural image.

    %Inputs:
    %structural_fn  - character array of the filename of the structural image.
    %spm_dir        - character array of the location of the spm directory.

    %Returns:
    %out    - struct with the following fields:
    %   - wm:       filename of the wm segmentation mask
    %   - gm:       filename of the gm segmentation mask
    %   - remove:   list of all the other files that are produced that can
    %               %safely be removed.                
    
    addpath(genpath(spm_dir))
    addpath(genpath(dcm2nii_dir))
    
    
    %First, create a .nii image from the dicom
    dcm2nii = strcat(dcm2nii_dir, '\dcm2niix');
    
    %Create a new directory for the .nii and the segmentation
    [direc, ~, ~]    = fileparts(structural_fn);
    newDir              = strcat(direc, filesep, 'spmSegmentation');
    mkdir(newDir);
    
    %Call the dcm2nii command
%     cmd = [dcm2nii ' -f "' fn '" -o "' newDir '"'];
    cmd = [dcm2nii ' -f %p_%s -o "' newDir '" "'  structural_fn '"'];
    system(cmd);
    
    
    %Remove any phase images
    phNii   = dir(fullfile(newDir, '*_ph.nii'));
    delete(fullfile(phNii.folder, phNii.name));
    
    phNii   = dir(fullfile(newDir, '*_pha.nii'));
    if ~isempty(phNii)
        delete(fullfile(phNii.folder, phNii.name));
    end
    
    phJson  = dir(fullfile(newDir, '*_ph.json'));
    delete(fullfile(phJson.folder, phJson.name));
    
    phJson  = dir(fullfile(newDir, '*_pha.json'));
    if ~isempty(phJson)
        delete(fullfile(phJson.folder, phJson.name));
    end
    
    aNii   = dir(fullfile(newDir, '*_a.nii'));
    if ~isempty(aNii)
        delete(fullfile(aNii.folder, aNii.name));
    end
    
    aJson   = dir(fullfile(newDir, '*_a.json'));
    if ~isempty(aNii)
        delete(fullfile(aJson.folder, aJson.name));
    end
    
    %Find the right .nii file:
    item   = dir(fullfile(newDir, '*.nii'));
    segment_fn      = fullfile(item.folder, item.name);
    
    %Next, perform the segmentation
    
    %Initiate
    spm('defaults','fmri');
    spm_jobman('initcfg');
    segmentation = struct;

    %Prepare parameters for segmentation
    % Channel
    segmentation.mb{1}.spm.spatial.preproc.channel.biasreg = 0.001;
    segmentation.mb{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    segmentation.mb{1}.spm.spatial.preproc.channel.write = [0 1];
    segmentation.mb{1}.spm.spatial.preproc.channel.vols = {segment_fn};
    % Tissue

    for t = 1:6
        segmentation.mb{1}.spm.spatial.preproc.tissue(t).tpm = {[spm_dir filesep 'tpm' filesep 'TPM.nii,' num2str(t)]};
        segmentation.mb{1}.spm.spatial.preproc.tissue(t).ngaus = t-1;
        segmentation.mb{1}.spm.spatial.preproc.tissue(t).native = [1 0];
        segmentation.mb{1}.spm.spatial.preproc.tissue(t).warped = [0 0];
    end
    % Warp
    segmentation.mb{1}.spm.spatial.preproc.warp.mrf = 1;
    segmentation.mb{1}.spm.spatial.preproc.warp.cleanup = 1;
    segmentation.mb{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    segmentation.mb{1}.spm.spatial.preproc.warp.affreg = 'mni';
    segmentation.mb{1}.spm.spatial.preproc.warp.fwhm = 0;
    segmentation.mb{1}.spm.spatial.preproc.warp.samp = 3;
    segmentation.mb{1}.spm.spatial.preproc.warp.write=[1 1];

    %
    % Run
    spm_jobman('run',segmentation.mb);

    % return filenames of all the created files
    [d, f, e] = fileparts(segment_fn);
    out = struct;
    out.remove = {  [d filesep 'y_' f e]    ...
                    [d filesep 'iy_' f e]   ...
                    [d filesep 'm' f e]     ...
                    [d filesep 'c3' f e]    ...
                    [d filesep 'c4' f e]    ...
                    [d filesep 'c5' f e]    ...
                    [d filesep 'c6' f e]    ...
                    [d filesep f '_seg8.mat']   };
    out.gm      = [d filesep 'c1' f e];      
    out.wm      = [d filesep 'c2' f e];

end

%%\