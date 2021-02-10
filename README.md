# SystemTools
Tools for CxSystem2 data science

These are additional tools for CxSystem2 repo data analysis and visualization. 
Modules starting with system_ include project independent tools.
Modules including word project, are project-specific.

The idea is to put project specific methods to project_XX.py. It's main class Project inherits
all the other module classes (SystemUtilities => SystemAnalysis => SystemViz => Project).

In the project_XX.py file, under the if name='__main__': set the project paths and variables.

When you instantiate a project object, you get all the inherited methods as project object attributes.
