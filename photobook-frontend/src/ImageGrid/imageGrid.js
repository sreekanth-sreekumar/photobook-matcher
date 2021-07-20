import './imageGrid.css';
import { useEffect, useState } from 'react';

const ImageGrid = (props) => {

    const [images, setImages] = useState([])
    const fetchData = () => {
        fetch('/get_images')
                .then(res => res.text())
                .then(data => {
                    const imageArray = data.split(' ');
                    setImages(imageArray)
                })
    }
    useEffect(() => {fetchData()}, [props.roundNr])
    return (
        <div className="img-grid">
            {images.map(img => <img alt={img} src={`${img}`} width="250" height="250"/>)}
        </div>
    )
}

export default ImageGrid;