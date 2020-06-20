# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 21:53:15 2020

@author: ryan_
"""

def convtranspose2d_output_shape(height_in, width_in, kernel_size, *,padding=0, stride=1,
                        output_padding=0, groups=1, dilation=1):
    

    height_out = (height_in-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding+1
    width_out = (height_in-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding+1
    
    print(f"HxW: {height_out}x{width_out}")
    
    return None
