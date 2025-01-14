# SYNCHRONOUS GENERATOR CAPABILITY CURVE GENERATOR
## Objective
This program was designed to create the capability curves of a synchronous generator given the machines' nominal parameters.

It comprises two functions:
1. **Capability Curve Generator:** generates the capability curve from the current limit curves for the rotor and stator
2. **Generator Capability Verification:** verify wheter a certain load condition can be achieved by a generator

## Requirements
The following is required for the calculation of curve intersections
https://github.com/sukhbinder/intersection

## References
This code was developed using the concepts and electric machinery modeling from Chapman's "Electric Machinery Fundamentals" (5th edition) and Uman's "Fitzgerald & Kingsley's Electric Machinery" (7th edition).

The sections used were, respectively, 4.11 (Synchronous Generator Ratings - Synchronous Generator Capability Curves, p.254) and 5.5 (Steady-state Operating Characteristics, p.293).

Along with several mathematical and programming ideas that will be cited when approriate throught the code.

## Developed by
- Felipe Baldner (@fbaldner)
- Júlia Avellar ()

## Modified by
- Jason Cheers (Edits in gen_capcurve.py, original jupter code unmodified)

## Version history
### Version 0.1 (2021-12-09):
- Initial release
- Defines generator input parameters and calculates both curves' data
- Plot both curves overlapping each other and prime-mover max power

### version 0.1mk (2022-8-15):
- Added basic calculations for the End Region Heating Limit
- Revised to build a composite curve with tripping regions for generator protection
- Added Generator Statistics on graph
- Results produced exceptable results in comparison to actual test data

## To-Do
- [ ] Code Clean-up
- [ ] Further Generator Example Testing
