void main() {
    // --- Codice omesso: Recupero dati dai G-buffer e codice specifico per tipo di luce ---
    for(uint i=0; i<lights.lightCount; i++){
        // Direzione luce e vettore view
        vec3 L = normalize(-lights.lights[i].direction);
        vec3 V = normalize(camPos - fragPos);
        vec3 H = normalize(V + L);
        // Fresnel Schlick
        vec3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - dot(H,V), 0.0, 1.0), 5.0);
        // Distribution GGX
        float a = roughness * roughness;
        float NdotH = max(dot(N,H), 0.0);
        float D = a * a / (3.14159 * pow(NdotH*NdotH*(a*a-1.0)+1.0, 2.0));
        // Geometry Schlick-GGX
        float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        float G = max(dot(N,V),0.0)/(max(dot(N,V),0.0)*(1.0-k)+k) *
                  max(dot(N,L),0.0)/(max(dot(N,L),0.0)*(1.0-k)+k);
        // Specular PBR
        vec3 spec = D * G * F / (4.0 * max(dot(N,V),0.0) * max(dot(N,L),0.0) + 0.001);
        // Diffuse PBR
        vec3 kD = (1.0 - F) * (1.0 - metallic);
        // Accumulo colore finale
        finalColor += (kD * albedo / 3.14159 + spec) *
                      lights.lights[i].color * lights.lights[i].intensity * max(dot(N,L),0.0);
    }
    // Ambient Occlusion
    finalColor += vec3(0.03) * albedo * ao;
    // Gamma Correction
    fragColor = vec4(pow(finalColor, vec3(1.0/2.2)), 1.0);
}
