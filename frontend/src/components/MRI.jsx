import React, { useState } from 'react';
import axios from 'axios';
import { Spinner, Alert, Button, Form, Row, Col, Card } from 'react-bootstrap';

const MRIModule = () => {
    // On gère un état pour chaque type de fichier
    const [files, setFiles] = useState({
        flair: null,
        t1: null,
        t1ce: null,
        t2: null,
        seg: null // La cible (Optionnelle)
    });
    
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    // Fonction générique pour mettre à jour le bon fichier
    const handleFileChange = (e, type) => {
        setFiles({
            ...files,
            [type]: e.target.files[0]
        });
    };

    const handleSegment = async () => {
        // Validation : Il faut au moins le FLAIR ou un autre
        if (!files.flair && !files.t1 && !files.t1ce && !files.t2) {
            setError("Veuillez uploader au moins une séquence IRM (ex: FLAIR).");
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        
        // On attache chaque fichier avec la CLÉ EXACTE attendue par le Python
        if (files.flair) formData.append('file_flair', files.flair);
        if (files.t1) formData.append('file_t1', files.t1);
        if (files.t1ce) formData.append('file_t1ce', files.t1ce);
        if (files.t2) formData.append('file_t2', files.t2);
        
        // Le fichier Cible (Target)
        if (files.seg) formData.append('file_seg', files.seg);

        try {
            // Note: Assure-toi que l'URL est bonne (localhost:5000 ou via proxy)
            const response = await axios.post('http://localhost:5000/api/mri', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setResult(response.data);
        } catch (err) {
            setError(err.response?.data?.error || "Erreur lors de la segmentation.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="mri-module">
            <h3 className="mb-4">Segmentation de Tumeur Cérébrale (BraTS)</h3>
            
            <Row className="mb-4">
                <Col md={6}>
                    <Card className="p-3 mb-3">
                        <Card.Title>1. Données du Patient (Inputs)</Card.Title>
                        <Form.Group className="mb-2">
                            <Form.Label>Séquence FLAIR (Obligatoire préférée)</Form.Label>
                            <Form.Control type="file" onChange={(e) => handleFileChange(e, 'flair')} accept=".nii,.nii.gz" />
                        </Form.Group>
                        <Form.Group className="mb-2">
                            <Form.Label>Séquence T1</Form.Label>
                            <Form.Control type="file" onChange={(e) => handleFileChange(e, 't1')} accept=".nii,.nii.gz" />
                        </Form.Group>
                        <Form.Group className="mb-2">
                            <Form.Label>Séquence T1ce (Contraste)</Form.Label>
                            <Form.Control type="file" onChange={(e) => handleFileChange(e, 't1ce')} accept=".nii,.nii.gz" />
                        </Form.Group>
                        <Form.Group className="mb-2">
                            <Form.Label>Séquence T2</Form.Label>
                            <Form.Control type="file" onChange={(e) => handleFileChange(e, 't2')} accept=".nii,.nii.gz" />
                        </Form.Group>
                    </Card>
                </Col>
                
                <Col md={6}>
                    <Card className="p-3 mb-3 border-success">
                        <Card.Title>2. Vérité Terrain (Cible)</Card.Title>
                        <Form.Group className="mb-2">
                            <Form.Label>Fichier de Segmentation (_seg.nii)</Form.Label>
                            <Form.Control type="file" onChange={(e) => handleFileChange(e, 'seg')} accept=".nii,.nii.gz" />
                            <Form.Text className="text-muted">Optionnel : Sert à comparer la prédiction avec la réalité.</Form.Text>
                        </Form.Group>
                    </Card>
                    
                    <div className="d-grid gap-2 mt-4">
                        <Button variant="primary" size="lg" onClick={handleSegment} disabled={loading}>
                            {loading ? <Spinner animation="border" size="sm" /> : 'Lancer l\'IA Multimodale'}
                        </Button>
                    </div>
                </Col>
            </Row>

            {error && <Alert variant="danger">{error}</Alert>}

            {result && (
                <Card className="mt-4 shadow-sm">
                    <Card.Body>
                        <h4 className="text-center">Résultat (Slice {result.slice_index})</h4>
                        <div className="d-flex justify-content-center gap-3 mt-3">
                            <div className="text-center">
                                <img src={`data:image/jpeg;base64,${result.mri_image}`} alt="MRI" style={{maxWidth: '200px'}} className="img-thumbnail" />
                                <p>Anatomie (FLAIR)</p>
                            </div>
                            <div className="text-center">
                                <img src={`data:image/png;base64,${result.segmentation_image}`} alt="Pred" style={{maxWidth: '200px'}} className="img-thumbnail bg-dark" />
                                <p>Prédiction IA</p>
                            </div>
                            {result.gt_image && (
                                <div className="text-center">
                                    <img src={`data:image/png;base64,${result.gt_image}`} alt="GT" style={{maxWidth: '200px'}} className="img-thumbnail bg-dark" />
                                    <p>Vérité Terrain</p>
                                </div>
                            )}
                        </div>
                    </Card.Body>
                </Card>
            )}
        </div>
    );
};

export default MRIModule;