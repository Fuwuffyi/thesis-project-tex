// Funzione per ricavare la posizione mondiale dal depth buffer
vec3 getWorldPos(vec2 uv, float depth) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc, depth, 1.0);
    vec4 viewPos = inverse(camera.proj) * clipPos;
    viewPos /= viewPos.w;
    vec4 worldPos = inverse(camera.view) * viewPos;
    return worldPos.xyz;
}

// Funzione Fresnel Schlick
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Distribution GGX
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

// Geometry Schlick-GGX
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Geometry Smith
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

// Main lighting pass
void main() {
    // --- Recupero dati dai G-buffer ---
    vec4 gNormalTex = texture(gNormal, fragUV);
    vec3 albedo   = texture(gAlbedo, fragUV).rgb;
    float roughness = gNormalTex.b;
    float metallic  = gNormalTex.a;
    float ao = texture(gAlbedo, fragUV).a;
    float depth = texture(gDepth, fragUV).r;
    vec3 worldPos = getWorldPos(fragUV, depth);
    vec3 N = normalize(decodeOctNormal(gNormalTex.xy));
    vec3 V = normalize(camera.viewPos - worldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 finalColor = vec3(0.0);
    // --- Loop sulle luci ---
    for (uint i = 0; i < lights.lightCount; ++i) {
        vec3 L;
        vec3 radiance = lights.lights[i].color * lights.lights[i].intensity;
        if (lights.lights[i].lightType == 0u) { // Directional
            L = normalize(-lights.lights[i].direction);
        } else { // Point / Spot
            L = normalize(lights.lights[i].position - worldPos);
            float dist = length(lights.lights[i].position - worldPos);
            float attenuation = 1.0 / (lights.lights[i].constant +
                                       lights.lights[i].linear * dist +
                                       lights.lights[i].quadratic * dist * dist);
            radiance *= attenuation;
            if (lights.lights[i].lightType == 2u) { // Spotlight
                float theta = dot(L, normalize(-lights.lights[i].direction));
                float epsilon = lights.lights[i].innerCone - lights.lights[i].outerCone;
                float intensity = clamp((theta - lights.lights[i].outerCone) / epsilon, 0.0, 1.0);
                radiance *= intensity;
            }
        }
        // --- PBR Shading ---
        vec3 H = normalize(V + L);
        float NDF = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
        vec3 specular = numerator / denominator;

        vec3 kS = F;
        vec3 kD = (1.0 - kS) * (1.0 - metallic);
        float NdotL = max(dot(N, L), 0.0);

        finalColor += (kD * albedo / PI + specular) * radiance * NdotL;
    }
    // --- Ambient Occlusion ---
    vec3 ambient = vec3(0.03) * albedo * ao;
    finalColor += ambient;
    // --- Gamma Correction ---
    float gamma = 2.2;
    fragColor = vec4(pow(finalColor, vec3(1.0 / gamma)), 1.0);
}
