from vitis.dsl import HLSProject
p = HLSProject(project='project_1', top='HLSKernel', solution='solution1', part='xck26-sfvc784-2LV-c')
p.add_file('top.cpp')
p.add_tb_file('host.cpp')
p.set_clock(period_ns=4.0)
p.csim()
p.csynth()
p.cosim()
p.export_xo(output='HLSKernel.xo')
