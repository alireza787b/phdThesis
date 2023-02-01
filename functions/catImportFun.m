function [cat] = catImportFun(filename)
%IMPORTFILE Import data from a text file
%  [TITLE, VAL1] = IMPORTFILE(FILENAME) reads data from text file
%  FILENAME for the default selection.  Returns the data as column
%  vectors.
%
%  [TITLE, VAL1] = IMPORTFILE(FILE, DATALINES) reads data for the
%  specified row interval(s) of text file FILENAME. Specify DATALINES as
%  a positive scalar integer or a N-by-2 array of positive scalar
%  integers for dis-contiguous row intervals.
%
%  Example:
%  [title, val1] = importfile("C:\Users\Alireza\OneDrive\matlabFatigue\logs\CAT_Alireza Ghaderi _2022-11-07_08-10-04.csv", [2, 38]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 19-Nov-2022 16:32:03

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [1, 38];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 9);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ";";

% Specify column names and types
opts.VariableNames = ["Var1", "title", "val1", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9"];
opts.SelectedVariableNames = ["title", "val1"];
opts.VariableTypes = ["string", "string", "double", "string", "string", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var1", "title", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "title", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "val1", "DecimalSeparator", ",");
opts = setvaropts(opts, "val1", "ThousandsSeparator", ".");

% Import the data
tbl = readtable(filename, opts);

%% Convert to output type
title = tbl.title;
val1 = tbl.val1;
raw = val1;
%cat.name = raw(2);
%cat.gender = raw(3);
%cat.birth = raw(4);
cat.omission_n = raw(16);
cat.omission_percent = raw(17);
cat.omission_percentile = raw(18);
%cat.omission_assesment = raw(19);
cat.comission_n = raw(20);
cat.comission_percent = raw(21);
cat.comission_percentile = raw(22);
%cat.comission_assessment = raw(23);
cat.rt_ms = raw(24);
cat.rt_percentile = raw(25);
%cat.rt_assessment = raw(26);
cat.rt_se_ms = raw(27);
cat.rt_se_percentile = raw(28);
%cat.rt_se_assessment = raw(29);
cat.variability = raw(30);
cat.detectibility = raw(31);
cat.responose_style = raw(32);
cat.perservation_n = raw(33);
cat.perservation_percentile = raw(34);
cat.block_change = raw(35);
cat.block_change_se = raw(36);
cat.isi_change = raw(37);
cat.isi_change_se = raw(38);


clear opts

%import blocks

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 9);

% Specify range and delimiter
opts.DataLines = [43, 59];
opts.Delimiter = ";";

% Specify column names and types
opts.VariableNames = ["Var1", "title", "block1", "block2", "block3", "block4", "block5", "block6", "Var9"];
opts.SelectedVariableNames = ["title", "block1", "block2", "block3", "block4", "block5", "block6"];
opts.VariableTypes = ["string", "string", "double", "double", "double", "double", "double", "double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var1", "title", "Var9"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "title", "Var9"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, ["block1", "block2", "block3", "block4", "block5", "block6"], "DecimalSeparator", ",");
opts = setvaropts(opts, ["block1", "block2", "block3", "block4", "block5", "block6"], "ThousandsSeparator", ".");

% Import the data
tbl = readtable(filename, opts);

%% Convert to output type
tbltitle = tbl.title;
tempBlock(:,1) = tbl.block1;
tempBlock(:,2) = tbl.block2;
tempBlock(:,3) = tbl.block3;
tempBlock(:,4) = tbl.block4;
tempBlock(:,5) = tbl.block5;
tempBlock(:,6) = tbl.block6;
for i=1:6
block(i).trials_n = tempBlock(1,i);
block(i).targets_n = tempBlock(2,i);
block(i).targets_percent = tempBlock(3,i);
block(i).none_targets_n = tempBlock(4,i);
block(i).none_targets_percent = tempBlock(5,i);
block(i).hits_n = tempBlock(6,i);
block(i).hits_percent = tempBlock(7,i);
block(i).omissions_n = tempBlock(8,i);
block(i).omissions_percent = tempBlock(9,i);
block(i).comissions_n = tempBlock(12,i);
block(i).comissions_percent = tempBlock(13,i);
block(i).rejections_n = tempBlock(10,i);
block(i).rejections_percent = tempBlock(11,i);
block(i).overal_rt_ms = tempBlock(14,i);
block(i).rt_ms = tempBlock(15,i);
block(i).comission_rt_ms = tempBlock(16,i);
block(i).rt_se_ms = tempBlock(17,i);
end


%% Clear temporary variables
clear opts tbl


cat.blocks = block;
end