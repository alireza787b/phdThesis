function [dirNames, dirPaths] = list_subfolders(directory)
    % Get all subdirectories of a directory, excluding '.' and '..'
    
    % Get list of all items in directory
    items = dir(directory);
    
    % Filter out non-directories and directories starting with '.'
    isDir = [items(:).isdir];
    name = {items(:).name};
    excludedDirs = ismember(name,{'.','..'}) | strncmp(name,'.',1);
    subDirs = items(isDir & ~excludedDirs);
    
    % Extract directory names and paths
    dirNames = {subDirs.name};
    nDirs = numel(dirNames);
    dirPaths = cell(1, nDirs);
    for i = 1:nDirs
        dirPaths{i} = fullfile(directory, dirNames{i});
    end
end
