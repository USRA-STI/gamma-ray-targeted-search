## Generalized Targeted Search

### Changes made during refactor
- InstrumentData class
  - Mainly an internal use class for the TargetedScanner to group relevant data/components together
  - Stores all necessary components of data for a single instrument for search
  - Centralizes the interfaces to access each data component individually (counts, background rates, and response, as 
well as spacecraft frames) and grouped (format data returns all necessary components for search)
  - Simplifies development of TargetedScanner class (known interfaces, simplified extraction of necessary values)
  - By grouping each relevant data component per instrument, we will reduce the occurrence of bugs where instrument data
may be mixed
  - Possible improvements:
    - Could be tied/have some hooks for InstrumentConfiguration, but currently it is being passed in where necessary
    - Need to implement format_data_by_reference for use in multi-mission search. This function should take in reference
data, such as the target sky position and the reference spacecraft's frame, compute the necessary values, and project
them as necessary
- TargetedScanner (TargetedSearch) class
  - Acts as the "orchestrator" of the search. Should be given at least one, or a set, of InstrumentData (through the 
add_instrument method), along with a SearchConfiguration. After this, the user should just call run_search to generate
the results objects. Internally creates and aligns timebins according to SearchConfiguration and the data's bins, and
uses InstrumentData class to generate the necessary data, then calculate the likelihood and derivative values to create 
results objects for each expected timebin
  - User can access functions to compute result data for single timebin (calculate_timebin_likelihood), or inspect the
timebins that are being used (get_timebins)
  - Improvements
    - Rename to TargetedSearch (fits with the expected naming convention from papers etc)
    - Implement components for multi-mission search (mainly stack_instrument_outputs)
    - Currently, hardcoded value for shape_data in calculate_timebin_likelihood. Should find a home for these in 
SearchConfiguration or InstrumentConfiguration classes.
    - Implement parallelization across timebins and optimize function where possible
- InstrumentConfiguration class
  - Stores the configuration details of a particular instrument
  - Attribute definitions allow simple extraction of derived values based on the core configuration where necessary in
TargetedScanner and InstrumentData classes, as well as on the user-side, for example when generating TTE/Phaii or 
BackgroundFitters objects
  - Automatically validates that expected attributes exist. Can be extended as other configuration details are 
implemented
- SearchConfiguration class
  - Stores the configuration details we want to use in the TargetedScanner class
  - Same benefit as above regarding attribute value extraction
  - Standardized set of configuration details; build_search_settings and validate_search_settings ensure that user has
provided required parameters for the search
- Response class
  - An abstract base class was created to specify the interface we need from the Response classes other missions will
develop, mainly just the load_response function
  - Each mission will develop their own subclass with their own implementation details, to be used along with counts and
background data in the InstrumentData class.

### Short Term

- Finalizing transition of gbm_targeted_search.py to use new class structures
  - Refactor base code, extract necessary changes from configuration_example.py
  - Implementing goodness-of-fit
    - Thoughts:
      - User implements goodness of fit functions in their workflow (as seen in configuration_example.py)
        - Standardize signature (parameters)
        - Input parameters can be changed as you see fit, I just thought counts and background rates would be the
standard necessary components. If other missions might use different parameters, another approach may be better, such 
as passing the data objects (InstrumentData) or other directly as input for the user to use, possibly passing in time
ranges as well.
      - Add validate_backgrounds function to TargetedScanner
      - Add validate_background function to InstrumentData
      - Add another configuration (probably in SearchConfiguration) for what time range with which to validate the 
background fit
      - For example (pseudocode):
```python
[InstrumentData]
def validate_background(tstart, tstop):
  counts, exposure = self.counts(tstart, tstop)
  background_rates, background_variance, good = self.background_rates(tstart, tstop, exposure)
  if(good.all() or good.nonzero() / good.size() > SearchConfiguration.threshold): # Add threshold to SearchConfiguration (user controlled)
      return True # Fit was good
  else:
      if(self.backup_fitters and self.backup_fitter_index < len(self.backup_fitters)) # Add in backup_fitter index attr in InstrumentData
          self.fitters = self.backup_fitters[self.backup_fitter_index] # Replace fitters; The first ones were bad
          self.backup_fitter_index += 1
          return self.validate_background(tstart, tstop) # Call again recursively
      else:
          return False # No backup fitters left, return
    
[TargetedScanner]
def validate_backgrounds():
    validation_tstart, validation_tstop = SearchConfiguration.get_validation_times() # Add func to SearchConfiguration
    for instrument in self.instrument_data:
        if(instrument.validate_background()):
            # Handle positive background fit (return True, probably)
        else:
            # Handle bad background fit (return False, probably) (if its this simple you dont need the if branch, just return
```
  - Implementing result filtration
    - Thoughts: 
      - Originally, I thought creating subclasses could be useful, but then the user would have to write classes to 
handle the extra optional values, which might be expecting too much
      - Add add_field function to Results class
      - Add filter function to Results class
      - For example (pseudocode):
```python
[Results Class]
def add_field(self, name, dtype, values=None):
    if name in self.data.dtype.names:
        raise ValueError(f"Field already exists in Results object")
    if values is None:
        values = np.zeros(self.data.shape[0], dtype=dtype)
    self.data = numpy.lib.recfunctions.append_fields(self.data, name, values, dtypes=dtype)
    self.optional_fields.add(name) # Create optional_fields attribute (as a set) to keep track of user added fields

def filter(self, predicate):
    mask = np.array([predicate(row) for row in self.data])
    filtered_results = Results(self.data[mask]) # Pseudocode; Create new result object extracting only truthy mask values
    return filtered_results
    
[User script]
def my_condition(row): # Standard signature; We are looking at each row of the structured array
    return row['pe'] > PE_THRESHOLD and row['template'] > 1

results # Previously computed from calling scanner.run_search()
pe_values = pe_calculator(scanner, results) # User defined function; Compute PE values using scanner object and results objects
# scanner parameter included as to extract internal data from InstrumentData classes (stored in scanner)
results.add_field('pe', 'f8', pe_values)
filtered_results = results.filter(my_condition)

# Can still access original results, along with their calculated PE values
# Can now access newly filtered results object, including added fields
```
  - Implementing plot generation
    - No comments at this time as I do not grasp the necessary components, whether results, internal data, etc

### Long Term
- Implementing multi-mission search
  - Thoughts:
    - For non-reference instruments, we can iterate over the Skygrid across all sky positions. The reference instrument
frame is static, then iterate across all skygrid positions. For each position, we will compute an offset of time it would take 
a signal to reach our spacecraft after or before the reference, and with this we can get the counts, backgrounds, and 
response data we would be getting from this sky position at a given time for the reference craft. My understanding here 
is dicey, but we would get a weighted average, or combine the outputs for each sky position in another way, or maybe we 
just have to extract portions that relate to that sky position. Data will have to be reprojected from spacecraft frame to 
reference, not sure what that entails. Then in calculate_timebin_likelihood we will stack the outputs of all instruments
    - Main function to handle multi-mission data formatting and extraction will be in InstrumentData (format_data_by_reference)
    - Main function to handle integration of that data when performing likelihood calculation will be in TargetedScanner
(inside calculate_timebin_likelihood the code should not require much modification)
      - Additional work is needed on stack_instrument_response (in TargetedScanner) to stack responses and ensure consistent
ordering of instrument inputs across counts, background rates, and response passed into like.calculate(*formatted_data)
    - Search Skygrid is passed as input to formatting functions so that they can match the skygrid of the reference craft.
Currently, skygrid is not a part of InstrumentData class other than as a parameter passed in from TargetedScanner. Could 
want to add in as part of InstrumentData in case resolutions don't match?
    - Currently the earthmask used for coinclr calculation is only extracted from reference spacecraft. Modification
required before implementing multi-mission search