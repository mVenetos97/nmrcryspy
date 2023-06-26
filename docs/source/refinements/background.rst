Background Information
======================

Why Perform a Structure Refinement
''''''''''''''''''''''''''''''''''

For crystalline solids, the gold standard of measuring a material’s structure is with
single crystal X-ray diffraction (XRD) measurements. Oftentimes, however, the single
crystal XRD measurement is unavailable. For example, many pharmaceutical molecules
and zeolites are only available in powder form and powder XRD is significantly less
accurate. For cases like these, we must use additional measurements, most commonly
solid-state NMR as NMR does not require perfect crystallinity. By combining NMR and
powder XRD measurements, along with quantum chemical calculations we can better refine
and elucidate the structures of difficult-to-measure materials.

How this Library Helps
''''''''''''''''''''''

Unfortunately, many workflows to do this structure refinement are extremely
computationally expensive, requiring days to weeks of compute time on state-of-the-art
supercomputer clusters. As a result, these methods are difficult to scale to high
throughput and can often only be used for small and simple materials.

In current methodologies, ab initio softwares are used to calculate the NMR
properties of interest. In order to obtain gradient information, however,
numerical derivatives must be calculated for each property and each coordinate.
All of these additional calculations quickly add up and as a result, refinements can
often require multiple days to complete. Additionally, due to the expense of these
computations, low levels of theory are typically used to speed up to refinements but
at the cost of introducing error in the final structure.

Machine learning (ML) may be a promising way to speed up these calculations as
performing a prediction is rapid compared to a full ab initio NMR calculation.
Furthermore, many modern machine learning codes, such as PyTorch, allow for the
automatic calculation of gradients (also known as autodifferentiation) while
performing a prediction. nmrcryspy uses this gradient information to rapidly
create the Jacobian matrix used in the refinement procedure and has been shown to
greatly reduce both the time required to do these refinement procedures as well as
the computational resources required.
