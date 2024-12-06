const dropArea = document.getElementById('drop-area');
const inputFile = document.getElementById('input-file');
const imageView = document.getElementById('img-view');
const imageLinkInput = document.getElementById('image-link');

// Handle file upload (via input or drag-and-drop)
inputFile.addEventListener("change", () => {
    clearImageLink();
    uploadImage();
});

function uploadImage() {
    if (inputFile.files && inputFile.files[0]) {
        const imgLink = URL.createObjectURL(inputFile.files[0]);
        setImagePreview(imgLink);
    }
}

// Handle drag-and-drop events
dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('dragover');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('dragover');
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('dragover');
    inputFile.files = e.dataTransfer.files;
    clearImageLink(); // Clear the link input
    uploadImage();
});

// Handle paste link functionality
imageLinkInput.addEventListener('input', () => {
    clearFileInput();
    const imgLink = imageLinkInput.value;
    if (imgLink) {
        setImagePreview(imgLink);
    }
});

// Set image preview
function setImagePreview(imgLink) {
    imageView.style.backgroundImage = `url(${imgLink})`;
    imageView.textContent = '';
    imageView.style.border = 'none';

    // Hide placeholder elements
    const icon = imageView.querySelector('.icon');
    const heading = imageView.querySelector('h3');
    const paragraph = imageView.querySelector('p');
    if (icon) icon.style.display = 'none';
    if (heading) heading.style.display = 'none';
    if (paragraph) paragraph.style.display = 'none';
}

// Clear file input when link is pasted
function clearFileInput() {
    inputFile.value = ''; // Reset file input
}

// Clear link input when a file is uploaded or dropped
function clearImageLink() {
    imageLinkInput.value = ''; // Reset link input
}
