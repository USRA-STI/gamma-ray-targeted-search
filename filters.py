# Copyright 2017-2022 by Universities Space Research Association (USRA). All rights reserved.
#
# Developed by: William Cleveland, Adam Goldstein, and Suman Bala
#               Universities Space Research Association
#               Science and Technology Institute
#               https://sti.usra.edu
#
# Developed by: Daniel Kocevski and Joshua Wood
#               National Aeronautics and Space Administration (NASA)
#               Marshall Space Flight Center
#               Astrophysics Branch (ST-12)
#
# Developed by: Lorenzo Scotton
#               University of Alabama in Huntsville
#               Center for Space Plasma and Aeronomic Research
#
# Very closely based on the gamma-ray burst targeted search (gbuts).
# Written by:
#               Lindy Blackburn
#               Center for Astrophysics (CfA) | Harvard & Smithsonian
#               https://github.com/lindyblackburn/gbuts
#
# Included in the generalized targeted search (gts) with permission from Lindy.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations under the
# License.
#
import numpy as np

def remove_pe(results, cr1=5, cr2=1, cr2thr=8):
    """Apply phosphorescence event (pe) veto and return a new Results object with the veto applied.
    
    Args:
        results (Results): The Results object to filter.
        cr1 (float): The threshold for pe_0/pe_1 ratio.
        cr2 (float): The threshold for pe_0/pe_2 ratio or pe_0 itself.
        cr2thr (float): The maximum value of pe_0 for vetoing.
    
    Returns:
        (Results): A new Results object with the veto applied.
    """
    if results.size == 0:
        return results

    icr1 = results['pe_0'] / np.maximum(0.1, results['pe_1']) < cr1
    icr2 = (results['pe_0'] / np.maximum(0.1, results['pe_2']) < cr2) | \
           (results['pe_0'] < cr2thr)

    filtered_data = results._data[(icr1 & icr2)]

    return Results.create(filtered_data, time_ref=results.time_ref, template_names=results.template_names)

def remove_dur_spec(results, dur, spec):
    """Remove results with matching duration and spectral template.

    Args:
        results (Results): The Results object to filter.
        dur (float): The duration in seconds to remove
        spec (int): Index of the spectral template to remove

    Returns:
        (Results): A new Results object without the duration + spectral template.
    """
    if results.size == 0:
        return results

    mask = (results['durations'] == dur) & (results['templates'] == spec)
    filtered_data = results._data[~mask]

    return Results.create(filtered_data, time_ref=results.time_ref, template_names=results.template_names)

def sky_cut(results, sky_diff=2):
    if results.size == 0:
        return results

    isky = (results['coinclr'] - results['loglr']) > sky_diff

    return Results.create(results[isky], time_ref=results.time_ref, templates_names=results.template_names)

def downselect(results, overlap_factor=0.2, threshold=None, combine_spec=True, 
               fixedwin=0, no_empty=False):
    if results.size == 0:
        return results
    
    if threshold:
        mask = (results['loglr'] >= threshold)
        if (mask.sum() == 0) and no_empty:
            mask = (results['loglr'] == results['loglr'].max())
        data = results._data[mask]
    else:
        data = results._data        
    
    unique_events = []
    sorted_events = data[(-data['loglr']).argsort()]
    
    for e1 in sorted_events:
        keep = True
        for e2 in unique_events:
            toverlap = min(e1['time'] + e1['duration'] / 2.0, e2['time'] + e2['duration'] / 2.0) \
                       - max(e1['time'] - e1['duration'] / 2.0, e2['time'] - e2['duration'] / 2.0) + fixedwin
            
            if (combine_spec or (e2['template'] == e1['template'])) and (toverlap > 0):
                amplitude = e1['snr_0'] / np.sqrt(e1['duration'])
                snr_expected = amplitude * toverlap / np.sqrt(e2['duration'])
                if e2['snr_0'] * overlap_factor < snr_expected:
                    keep = False
                    break
        if keep:
            unique_events.append(e1)
    
    data = np.array(unique_events, dtype=results.dtype)

    return Results.create(data, time_ref=results.time_ref, template_names=results.template_names)
