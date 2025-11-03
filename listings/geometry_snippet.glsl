// Tangent-bitangent-normal matrix
mat3 computeTBN(vec3 N, vec2 uv, vec3 pos) {
   vec3 dp1 = dFdx(pos);
   vec3 dp2 = dFdy(pos);
   vec2 duv1 = dFdx(uv);
   vec2 duv2 = dFdy(uv);
   vec3 T = normalize(duv2.y * dp1 - duv1.y * dp2);
   vec3 B = normalize(-duv2.x * dp1 + duv1.x * dp2);
   return mat3(T, B, N);
}

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
