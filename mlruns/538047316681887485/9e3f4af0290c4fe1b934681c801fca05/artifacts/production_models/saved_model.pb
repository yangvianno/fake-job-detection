��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.19.02v2.19.0-rc0-6-ge36baa302928��
�
output/biasVarHandleOp*
_output_shapes
: *

debug_nameoutput/bias/*
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
�
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape:	�N@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�N@*
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
�
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape
:@ *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@ *
dtype0
�
output/kernelVarHandleOp*
_output_shapes
: *

debug_nameoutput/kernel/*
dtype0*
shape
: *
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

: *
dtype0
�
output/bias_1VarHandleOp*
_output_shapes
: *

debug_nameoutput/bias_1/*
dtype0*
shape:*
shared_nameoutput/bias_1
k
!output/bias_1/Read/ReadVariableOpReadVariableOpoutput/bias_1*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpoutput/bias_1*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
output/kernel_1VarHandleOp*
_output_shapes
: * 

debug_nameoutput/kernel_1/*
dtype0*
shape
: * 
shared_nameoutput/kernel_1
s
#output/kernel_1/Read/ReadVariableOpReadVariableOpoutput/kernel_1*
_output_shapes

: *
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpoutput/kernel_1*
_class
loc:@Variable_1*
_output_shapes

: *
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
: *
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

: *
dtype0
�
%seed_generator_1/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_1/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_1/seed_generator_state
�
9seed_generator_1/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_1/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_2/Initializer/ReadVariableOpReadVariableOp%seed_generator_1/seed_generator_state*
_class
loc:@Variable_2*
_output_shapes
:*
dtype0	
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0	*
shape:*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0	
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
�
dense_1/bias_1VarHandleOp*
_output_shapes
: *

debug_namedense_1/bias_1/*
dtype0*
shape: *
shared_namedense_1/bias_1
m
"dense_1/bias_1/Read/ReadVariableOpReadVariableOpdense_1/bias_1*
_output_shapes
: *
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpdense_1/bias_1*
_class
loc:@Variable_3*
_output_shapes
: *
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape: *
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
e
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
�
dense_1/kernel_1VarHandleOp*
_output_shapes
: *!

debug_namedense_1/kernel_1/*
dtype0*
shape
:@ *!
shared_namedense_1/kernel_1
u
$dense_1/kernel_1/Read/ReadVariableOpReadVariableOpdense_1/kernel_1*
_output_shapes

:@ *
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpdense_1/kernel_1*
_class
loc:@Variable_4*
_output_shapes

:@ *
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape
:@ *
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
i
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes

:@ *
dtype0
�
#seed_generator/seed_generator_stateVarHandleOp*
_output_shapes
: *4

debug_name&$seed_generator/seed_generator_state/*
dtype0	*
shape:*4
shared_name%#seed_generator/seed_generator_state
�
7seed_generator/seed_generator_state/Read/ReadVariableOpReadVariableOp#seed_generator/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_5/Initializer/ReadVariableOpReadVariableOp#seed_generator/seed_generator_state*
_class
loc:@Variable_5*
_output_shapes
:*
dtype0	
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0	*
shape:*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0	
e
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
:*
dtype0	
�
dense/bias_1VarHandleOp*
_output_shapes
: *

debug_namedense/bias_1/*
dtype0*
shape:@*
shared_namedense/bias_1
i
 dense/bias_1/Read/ReadVariableOpReadVariableOpdense/bias_1*
_output_shapes
:@*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpdense/bias_1*
_class
loc:@Variable_6*
_output_shapes
:@*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:@*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
:@*
dtype0
�
dense/kernel_1VarHandleOp*
_output_shapes
: *

debug_namedense/kernel_1/*
dtype0*
shape:	�N@*
shared_namedense/kernel_1
r
"dense/kernel_1/Read/ReadVariableOpReadVariableOpdense/kernel_1*
_output_shapes
:	�N@*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpdense/kernel_1*
_class
loc:@Variable_7*
_output_shapes
:	�N@*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:	�N@*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
j
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
:	�N@*
dtype0
v
serve_tfidf_inputPlaceholder*(
_output_shapes
:����������N*
dtype0*
shape:����������N
�
StatefulPartitionedCallStatefulPartitionedCallserve_tfidf_inputdense/kernel_1dense/bias_1dense_1/kernel_1dense_1/bias_1output/kernel_1output/bias_1*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU 2J 8� �J *5
f0R.
,__inference_signature_wrapper___call___43757
�
serving_default_tfidf_inputPlaceholder*(
_output_shapes
:����������N*
dtype0*
shape:����������N
�
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_tfidf_inputdense/kernel_1dense/bias_1dense_1/kernel_1dense_1/bias_1output/kernel_1output/bias_1*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU 2J 8� �J *5
f0R.
,__inference_signature_wrapper___call___43774

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*
<
0
	1

2
3
4
5
6
7*
.
0
	1
2
3
4
5*


0
1*
.
0
1
2
3
4
5*
* 

trace_0* 
"
	serve
serving_default* 
JD
VARIABLE_VALUE
Variable_7&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_6&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_5&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_4&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_3&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_2&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
Variable_1&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
HB
VARIABLE_VALUEVariable&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEoutput/kernel_1+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_1/kernel_1+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEdense/bias_1+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense/kernel_1+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_1/bias_1+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEoutput/bias_1+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variableoutput/kernel_1dense_1/kernel_1dense/bias_1dense/kernel_1dense_1/bias_1output/bias_1Const*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *'
f"R 
__inference__traced_save_43914
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variableoutput/kernel_1dense_1/kernel_1dense/bias_1dense/kernel_1dense_1/bias_1output/bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J **
f%R#
!__inference__traced_restore_43965��
�l
�
__inference__traced_save_43914
file_prefix4
!read_disablecopyonread_variable_7:	�N@1
#read_1_disablecopyonread_variable_6:@1
#read_2_disablecopyonread_variable_5:	5
#read_3_disablecopyonread_variable_4:@ 1
#read_4_disablecopyonread_variable_3: 1
#read_5_disablecopyonread_variable_2:	5
#read_6_disablecopyonread_variable_1: /
!read_7_disablecopyonread_variable::
(read_8_disablecopyonread_output_kernel_1: ;
)read_9_disablecopyonread_dense_1_kernel_1:@ 4
&read_10_disablecopyonread_dense_bias_1:@;
(read_11_disablecopyonread_dense_kernel_1:	�N@6
(read_12_disablecopyonread_dense_1_bias_1: 5
'read_13_disablecopyonread_output_bias_1:
savev2_const
identity_29��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: d
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_7*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_7^Read/DisableCopyOnRead*
_output_shapes
:	�N@*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	�N@b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�N@h
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_variable_6*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_variable_6^Read_1/DisableCopyOnRead*
_output_shapes
:@*
dtype0Z

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@h
Read_2/DisableCopyOnReadDisableCopyOnRead#read_2_disablecopyonread_variable_5*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp#read_2_disablecopyonread_variable_5^Read_2/DisableCopyOnRead*
_output_shapes
:*
dtype0	Z

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0	*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0	*
_output_shapes
:h
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_variable_4*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_variable_4^Read_3/DisableCopyOnRead*
_output_shapes

:@ *
dtype0^

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes

:@ c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:@ h
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_variable_3*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_variable_3^Read_4/DisableCopyOnRead*
_output_shapes
: *
dtype0Z

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: h
Read_5/DisableCopyOnReadDisableCopyOnRead#read_5_disablecopyonread_variable_2*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp#read_5_disablecopyonread_variable_2^Read_5/DisableCopyOnRead*
_output_shapes
:*
dtype0	[
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0	*
_output_shapes
:h
Read_6/DisableCopyOnReadDisableCopyOnRead#read_6_disablecopyonread_variable_1*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp#read_6_disablecopyonread_variable_1^Read_6/DisableCopyOnRead*
_output_shapes

: *
dtype0_
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes

: e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

: f
Read_7/DisableCopyOnReadDisableCopyOnRead!read_7_disablecopyonread_variable*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp!read_7_disablecopyonread_variable^Read_7/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:m
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_output_kernel_1*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_output_kernel_1^Read_8/DisableCopyOnRead*
_output_shapes

: *
dtype0_
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes

: e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

: n
Read_9/DisableCopyOnReadDisableCopyOnRead)read_9_disablecopyonread_dense_1_kernel_1*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp)read_9_disablecopyonread_dense_1_kernel_1^Read_9/DisableCopyOnRead*
_output_shapes

:@ *
dtype0_
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes

:@ e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:@ l
Read_10/DisableCopyOnReadDisableCopyOnRead&read_10_disablecopyonread_dense_bias_1*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp&read_10_disablecopyonread_dense_bias_1^Read_10/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@n
Read_11/DisableCopyOnReadDisableCopyOnRead(read_11_disablecopyonread_dense_kernel_1*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp(read_11_disablecopyonread_dense_kernel_1^Read_11/DisableCopyOnRead*
_output_shapes
:	�N@*
dtype0a
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
:	�N@f
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:	�N@n
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_dense_1_bias_1*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_dense_1_bias_1^Read_12/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: m
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_output_bias_1*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_output_bias_1^Read_13/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_28Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_29IdentityIdentity_28:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_29Identity_29:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:-)
'
_user_specified_nameoutput/bias_1:.*
(
_user_specified_namedense_1/bias_1:.*
(
_user_specified_namedense/kernel_1:,(
&
_user_specified_namedense/bias_1:0
,
*
_user_specified_namedense_1/kernel_1:/	+
)
_user_specified_nameoutput/kernel_1:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
,__inference_signature_wrapper___call___43757
tfidf_input
unknown:	�N@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltfidf_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU 2J 8� �J *#
fR
__inference___call___43739o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������N: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name43753:%!

_user_specified_name43751:%!

_user_specified_name43749:%!

_user_specified_name43747:%!

_user_specified_name43745:%!

_user_specified_name43743:U Q
(
_output_shapes
:����������N
%
_user_specified_nametfidf_input
�#
�
__inference___call___43739
tfidf_inputC
0mlp_model_1_dense_1_cast_readvariableop_resource:	�N@A
3mlp_model_1_dense_1_biasadd_readvariableop_resource:@D
2mlp_model_1_dense_1_2_cast_readvariableop_resource:@ C
5mlp_model_1_dense_1_2_biasadd_readvariableop_resource: C
1mlp_model_1_output_1_cast_readvariableop_resource: >
0mlp_model_1_output_1_add_readvariableop_resource:
identity��*mlp_model_1/dense_1/BiasAdd/ReadVariableOp�'mlp_model_1/dense_1/Cast/ReadVariableOp�,mlp_model_1/dense_1_2/BiasAdd/ReadVariableOp�)mlp_model_1/dense_1_2/Cast/ReadVariableOp�'mlp_model_1/output_1/Add/ReadVariableOp�(mlp_model_1/output_1/Cast/ReadVariableOp�
'mlp_model_1/dense_1/Cast/ReadVariableOpReadVariableOp0mlp_model_1_dense_1_cast_readvariableop_resource*
_output_shapes
:	�N@*
dtype0�
mlp_model_1/dense_1/MatMulMatMultfidf_input/mlp_model_1/dense_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*mlp_model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp3mlp_model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
mlp_model_1/dense_1/BiasAddBiasAdd$mlp_model_1/dense_1/MatMul:product:02mlp_model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
mlp_model_1/dense_1/ReluRelu$mlp_model_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
)mlp_model_1/dense_1_2/Cast/ReadVariableOpReadVariableOp2mlp_model_1_dense_1_2_cast_readvariableop_resource*
_output_shapes

:@ *
dtype0�
mlp_model_1/dense_1_2/MatMulMatMul&mlp_model_1/dense_1/Relu:activations:01mlp_model_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
,mlp_model_1/dense_1_2/BiasAdd/ReadVariableOpReadVariableOp5mlp_model_1_dense_1_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
mlp_model_1/dense_1_2/BiasAddBiasAdd&mlp_model_1/dense_1_2/MatMul:product:04mlp_model_1/dense_1_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� |
mlp_model_1/dense_1_2/ReluRelu&mlp_model_1/dense_1_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
(mlp_model_1/output_1/Cast/ReadVariableOpReadVariableOp1mlp_model_1_output_1_cast_readvariableop_resource*
_output_shapes

: *
dtype0�
mlp_model_1/output_1/MatMulMatMul(mlp_model_1/dense_1_2/Relu:activations:00mlp_model_1/output_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'mlp_model_1/output_1/Add/ReadVariableOpReadVariableOp0mlp_model_1_output_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
mlp_model_1/output_1/AddAddV2%mlp_model_1/output_1/MatMul:product:0/mlp_model_1/output_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
mlp_model_1/output_1/SigmoidSigmoidmlp_model_1/output_1/Add:z:0*
T0*'
_output_shapes
:���������o
IdentityIdentity mlp_model_1/output_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^mlp_model_1/dense_1/BiasAdd/ReadVariableOp(^mlp_model_1/dense_1/Cast/ReadVariableOp-^mlp_model_1/dense_1_2/BiasAdd/ReadVariableOp*^mlp_model_1/dense_1_2/Cast/ReadVariableOp(^mlp_model_1/output_1/Add/ReadVariableOp)^mlp_model_1/output_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������N: : : : : : 2X
*mlp_model_1/dense_1/BiasAdd/ReadVariableOp*mlp_model_1/dense_1/BiasAdd/ReadVariableOp2R
'mlp_model_1/dense_1/Cast/ReadVariableOp'mlp_model_1/dense_1/Cast/ReadVariableOp2\
,mlp_model_1/dense_1_2/BiasAdd/ReadVariableOp,mlp_model_1/dense_1_2/BiasAdd/ReadVariableOp2V
)mlp_model_1/dense_1_2/Cast/ReadVariableOp)mlp_model_1/dense_1_2/Cast/ReadVariableOp2R
'mlp_model_1/output_1/Add/ReadVariableOp'mlp_model_1/output_1/Add/ReadVariableOp2T
(mlp_model_1/output_1/Cast/ReadVariableOp(mlp_model_1/output_1/Cast/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:U Q
(
_output_shapes
:����������N
%
_user_specified_nametfidf_input
�A
�
!__inference__traced_restore_43965
file_prefix.
assignvariableop_variable_7:	�N@+
assignvariableop_1_variable_6:@+
assignvariableop_2_variable_5:	/
assignvariableop_3_variable_4:@ +
assignvariableop_4_variable_3: +
assignvariableop_5_variable_2:	/
assignvariableop_6_variable_1: )
assignvariableop_7_variable:4
"assignvariableop_8_output_kernel_1: 5
#assignvariableop_9_dense_1_kernel_1:@ .
 assignvariableop_10_dense_bias_1:@5
"assignvariableop_11_dense_kernel_1:	�N@0
"assignvariableop_12_dense_1_bias_1: /
!assignvariableop_13_output_bias_1:
identity_15��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_7Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_6Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_5Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_4Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_3Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_2Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variableIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_output_kernel_1Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_1_kernel_1Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_bias_1Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_kernel_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_bias_1Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_output_bias_1Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_15Identity_15:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
: : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:-)
'
_user_specified_nameoutput/bias_1:.*
(
_user_specified_namedense_1/bias_1:.*
(
_user_specified_namedense/kernel_1:,(
&
_user_specified_namedense/bias_1:0
,
*
_user_specified_namedense_1/kernel_1:/	+
)
_user_specified_nameoutput/kernel_1:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
,__inference_signature_wrapper___call___43774
tfidf_input
unknown:	�N@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltfidf_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU 2J 8� �J *#
fR
__inference___call___43739o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������N: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%!

_user_specified_name43770:%!

_user_specified_name43768:%!

_user_specified_name43766:%!

_user_specified_name43764:%!

_user_specified_name43762:%!

_user_specified_name43760:U Q
(
_output_shapes
:����������N
%
_user_specified_nametfidf_input"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
:
tfidf_input+
serve_tfidf_input:0����������N<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict*�
serving_default�
D
tfidf_input5
serving_default_tfidf_input:0����������N>
output_02
StatefulPartitionedCall_1:0���������tensorflow/serving/predict:�
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
X
0
	1

2
3
4
5
6
7"
trackable_list_wrapper
J
0
	1
2
3
4
5"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trace_02�
__inference___call___43739�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *+�(
&�#
tfidf_input����������Nztrace_0
7
	serve
serving_default"
signature_map
:	�N@2dense/kernel
:@2
dense/bias
/:-	2#seed_generator/seed_generator_state
 :@ 2dense_1/kernel
: 2dense_1/bias
1:/	2%seed_generator_1/seed_generator_state
: 2output/kernel
:2output/bias
: 2output/kernel
 :@ 2dense_1/kernel
:@2
dense/bias
:	�N@2dense/kernel
: 2dense_1/bias
:2output/bias
�B�
__inference___call___43739tfidf_input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_signature_wrapper___call___43757tfidf_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
  

kwonlyargs�
jtfidf_input
kwonlydefaults
 
annotations� *
 
�B�
,__inference_signature_wrapper___call___43774tfidf_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
  

kwonlyargs�
jtfidf_input
kwonlydefaults
 
annotations� *
 �
__inference___call___43739b	5�2
+�(
&�#
tfidf_input����������N
� "!�
unknown����������
,__inference_signature_wrapper___call___43757�	D�A
� 
:�7
5
tfidf_input&�#
tfidf_input����������N"3�0
.
output_0"�
output_0����������
,__inference_signature_wrapper___call___43774�	D�A
� 
:�7
5
tfidf_input&�#
tfidf_input����������N"3�0
.
output_0"�
output_0���������