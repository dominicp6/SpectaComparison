import os
from collections import defaultdict
from math import floor

import numpy as np
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

def get_experimental_and_computational_spectra(data_dir, delimiter=" ") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Get experimental and computational spectra from a directory.
    Assumes experiment spectrum ends with "dpt" and computational spectra end with "tab".
    """
    experimental_spectrum = None
    computational_spectra = defaultdict(lambda: None)
    for filename in os.listdir(data_dir):
        if filename.endswith("tab"):
            dataseries_name = filename.split(".tab")[0]
            dataseries_spectrum = np.loadtxt(os.path.join(data_dir, filename), delimiter=delimiter)
            computational_spectra[dataseries_name] = dataseries_spectrum
        elif filename.endswith("dpt"):
            experimental_spectrum = np.loadtxt(os.path.join(data_dir, filename), delimiter=delimiter)
        else:
            print(f"Unexpected file: {filename}")

    return experimental_spectrum, computational_spectra

def get_wavenumber_bounds(spectra, reference_spectrum = None) -> tuple[float, float]:
    """
    Finds the smallest and largest wavenumbers across all spectra.
    """
    if reference_spectrum is not None:
        ref_wavenumber_min = reference_spectrum[:, 0].min()
        ref_wavenumber_max = reference_spectrum[:, 0].max()
        global_wavenumber_max_min = max([max([spectrum[:, 0].min() for spectrum in spectra.values()]), ref_wavenumber_min])
        global_wavenumber_min_max = min([min([spectrum[:, 0].max() for spectrum in spectra.values()]), ref_wavenumber_max])
    else:
        global_wavenumber_max_min = max([spectrum[:, 0].min() for spectrum in spectra.values()])
        global_wavenumber_min_max = min([spectrum[:, 0].max() for spectrum in spectra.values()])

    return global_wavenumber_max_min, global_wavenumber_min_max

def get_standard_wavenumbers(spectra, min_wavenumber, max_wavenumber, reference_spectrum = None) -> np.ndarray:
    """
    Generates a standard set of wavenumbers using the finest resolution of all spectra.
    min_wavenumber and max_wavenumber are the bounds of the wavenumber range to consider.
    """
    if reference_spectrum is not None:
        finest_resolution = min(min([np.abs(np.diff(spectrum[:, 0])).min() for spectrum in spectra.values()]), np.abs(np.diff(reference_spectrum[:, 0])).min())
    else:
        finest_resolution = min([np.abs(np.diff(spectrum[:, 0])).min() for spectrum in spectra.values()])
    
    length_of_spectrum_vector = floor((max_wavenumber - min_wavenumber) / finest_resolution)
    standard_wavenumbers = np.linspace(min_wavenumber, max_wavenumber, length_of_spectrum_vector)

    return standard_wavenumbers

def standardise_spectrum(spectrum: np.ndarray, standard_wavenumbers: np.ndarray, standardisation_method="highest_peak") -> np.ndarray:
    """
    Adjusts the spectrum so that it is defined only at standard_wavenumbers, then applies a standardisation method.
    """

    interpolation_function = interp1d(spectrum[:,0], spectrum[:,1], kind='linear')
    interpolated_intensity = interpolation_function(standard_wavenumbers)

    if standardisation_method == "highest_peak":
        # Normalise to have maximum intensity of 1
        interpolated_intensity = interpolated_intensity / interpolated_intensity.max()
    elif standardisation_method == "integral":
        # Normalise to have a total integral of 1
        interpolated_intensity = interpolated_intensity / np.trapz(interpolated_intensity, standard_wavenumbers)
    else:
        raise NotImplementedError

    # Construct the spectrum array (wavenumbers, intensities)
    standardised_spectrum = np.column_stack((standard_wavenumbers, interpolated_intensity))

    return standardised_spectrum

def standardise_spectra_with_reference(spectra: dict[str, np.ndarray], reference_spectrum: np.ndarray, standardisation_method="highest_peak") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Standardises the reference and computational spectra to a common set of wavenumbers and a common intensity scale.
    """

    # Find the lowest and largest wavenumbers across all spectra
    min_wavenumber, max_wavenumber = get_wavenumber_bounds(spectra, reference_spectrum=reference_spectrum)
    # Generate a standard set of wavenumbers using the finest resolution of all spectra
    standard_wavenumbers = get_standard_wavenumbers(spectra, min_wavenumber, max_wavenumber, reference_spectrum=reference_spectrum)

    # Standardise the reference spectrum
    standardised_reference_spectrum = standardise_spectrum(reference_spectrum, standard_wavenumbers, standardisation_method)

    standardised_spectra = {}
    for spectrum_name, spectrum in spectra.items():
        # Standardise each spectrum
        standardised_spectrum = standardise_spectrum(spectrum, standard_wavenumbers, standardisation_method)

        # Store the standardised spectrum
        standardised_spectra[spectrum_name] = standardised_spectrum

    return standardised_reference_spectrum, standardised_spectra

def standardise_spectra(spectra: dict[str, np.ndarray], standardisation_method="highest_peak") -> dict[str, np.ndarray]:
    """
    Standardises spectra to a common set of wavenumbers and a common intensity scale.
    """
    # Find the lowest and largest wavenumbers across all spectra
    min_wavenumber, max_wavenumber = get_wavenumber_bounds(spectra)
    # Generate a standard set of wavenumbers using the finest resolution of all spectra
    standard_wavenumbers = get_standard_wavenumbers(spectra, min_wavenumber, max_wavenumber)

    standardised_spectra = {}
    for spectrum_name, spectrum in spectra.items():
        # Standardise each spectrum
        standardised_spectrum = standardise_spectrum(spectrum, standard_wavenumbers, standardisation_method)

        # Store the standardised spectrum
        standardised_spectra[spectrum_name] = standardised_spectrum

    return standardised_spectra

def fit_PCA_distances(spectra: dict[str, np.ndarray], d=10) -> tuple[PCA, np.ndarray]:
    """
    Fit a PCA model to the space of spectra.
    """
    # Create a PCA instance
    pca = PCA(n_components=d)

    # Fit the PCA model to the space of experimental and theoretical spectra
    pca.fit(np.vstack([spectrum[:,1] for spectrum in spectra]))

    explained_variance = pca.explained_variance_ratio_

    return pca, explained_variance

def vector_Lp_distance(vector1, vector2, p=1):
    """
    Calculates the Lp distance between two vectors.
    """
    return np.sum(np.abs(vector1 - vector2)**p)**(1/p)

def consine_distance(vector1, vector2):
    """
    Calculates the cosine distance between two vectors.
    """
    vector1 = np.squeeze(vector1)  # Ensure vector1 is a 1D array
    vector2 = np.squeeze(vector2)  # Ensure vector2 is a 1D array
    cosine_dist = (1 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))/2
    return cosine_dist

def compute_PCA_distance(spectrum1: np.ndarray, spectrum2: np.ndarray, pca: PCA, p=1, cosine_dist = False) -> float:
    """
    Compute the distance between two spectra in PCA space.
    """

    # Transform the spectra into PCA space
    spectrum1_pca = pca.transform([spectrum1[:, 1]])
    spectrum2_pca = pca.transform([spectrum2[:, 1]])

    # Compute the Lp distance between the spectra in PCA space
    if cosine_dist:
        dist = consine_distance(spectrum1_pca, spectrum2_pca)
    else:
        dist = vector_Lp_distance(spectrum1_pca, spectrum2_pca, p=p)
    
    return dist

def PCA_analysis(all_spectra: dict[str, np.ndarray], 
                 spectra_minus_reference: dict[str, np.ndarray], 
                 reference_spectrum: np.ndarray, 
                 d:int=10, 
                 p:float=1, 
                 cosine_dist:bool = False) -> tuple[dict[str, float], np.ndarray]:
    """
    Perform PCA analysis on the spectra and compute the PCA distances between the reference spectrum and the other spectra.
    """
    pca, explained_variance = fit_PCA_distances(all_spectra, d=d)

    PCA_distances = {}
    for spectrum_name, spectrum in spectra_minus_reference.items():
        PCA_distances[spectrum_name] = compute_PCA_distance(reference_spectrum, spectrum, pca, p=p, cosine_dist=cosine_dist)

    # sort by distance
    PCA_distances = dict(sorted(PCA_distances.items(), key=lambda item: item[1]))

    return PCA_distances, explained_variance