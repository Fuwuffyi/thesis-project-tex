// OCT-Encoding
vec2 encodeOctNormal(vec3 n) {
   n /= (abs(n.x) + abs(n.y) + abs(n.z));
   vec2 enc = n.xy;
   if (n.z < 0.0) {
      enc = (1.0 - abs(enc.yx)) * sign(enc.xy);
   }
   return enc * 0.5 + 0.5;
}

void main() {
   mat3 TBN = computeTBN(normalize(fragNormal), fragUV, fragPos);
   vec3 viewDir = normalize(camera.viewPos - fragPos);
   vec3 normalTS = texture(normalSampler, fragUV).rgb * 2.0 - 1.0;
   vec3 normalWS = normalize(TBN * normalTS);
   gAlbedo = vec4(texture(albedoSampler, fragUV).rgb * material.albedo, texture(aoSampler, fragUV).r * material.ao);
   gNormal = vec4(encodeOctNormal(normalWS), texture(roughnessSampler, fragUV).r * material.roughness,
      texture(metallicSampler, fragUV).r * material.metallic);
}
