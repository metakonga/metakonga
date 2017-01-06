/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#define STRINGIFY(A) #A

// vertex shader
const char *vertexShader = STRINGIFY(
	uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels
uniform float densityScale;
uniform float densityOffset;
void main()
{
	// calculate window-space point size
	vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
	float dist = length(posEye);
	gl_PointSize = gl_Vertex.w * (pointScale / dist);

	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

	gl_FrontColor = gl_Color;
}
);

// pixel shader for rendering points as shaded spheres
const char *spherePixelShader = STRINGIFY(
	void main()
{
	const vec3 lightDir = vec3(0.577, 0.577, 0.577);

	// calculate normal from texture coordinates
	vec3 N;
	N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
	float mag = dot(N.xy, N.xy);
	if (mag > 1.0) discard;   // kill pixels outside circle
	N.z = sqrt(1.0 - mag);

	// calculate lighting
	float diffuse = max(0.0, dot(lightDir, N));

	gl_FragColor = gl_Color * diffuse;
}
);

const char *fluidRenderParticleVertex = STRINGIFY(
	uniform vec2 screenSize;
out vec3 eyespacePos;
out float eyespaceRadius;
out float velocity;
void main()
{
	// Transform
	vec4 eyespacePos4 = vec4(gl_Vertex.xyz, 1.0f) * gl_ModelViewMatrix;
	eyespacePos = eyespacePos4.xyz;
	eyespaceRadius = 1.0f / (-eyespacePos.z * 4.0f * (1.0f / screenSize.y));
	vec4 clipspacePos = eyespacePos4 * glProjectionMatrix;

	// Set up variables for rasterizer
	gl_Position = clipspacePos;
	gl_PointSize = eyespaceRadius;

	// Send velocity to fragment shader
	velocity = gl_Vertex.w;
}
);

const char *fluidRenderParticleDepthFragment = STRINGIFY(
	// Particle depth fragment shader

	// Parameters from the vertex shader
	in vec3 eyespacePos;
in float eyespaceRadius;

// Uniforms
uniform vec2 screenSize;

// Textures
// uniform sampler2D terrainTexture;

// Output
out float particleDepth;

void main() {
	vec3 normal;

	// See where we are inside the point sprite
	normal.xy = (gl_PointCoord - 0.5f) * 2.0f;
	float dist = length(normal);

	// Outside sphere? Discard.
	if (dist > 1.0f) {
		discard;
	}

	// Set up rest of normal
	normal.z = sqrt(1.0f - dist);
	normal.y = -normal.y;
	normal = normalize(normal);

	// Calculate fragment position in eye space, project to find depth
	vec4 fragPos = vec4(eyespacePos + normal * eyespaceRadius / screenSize.y, 1.0);
	vec4 clipspacePos = fragPos * gl_ProjectionMatrix;

	// Set up output
	float far = gl_DepthRange.far;
	float near = gl_DepthRange.near;
	float deviceDepth = clipspacePos.z / clipspacePos.w;
	float fragDepth = (((far - near) * deviceDepth) + near + far) / 2.0;
	gl_FragDepth = fragDepth;

	// 	if(fragDepth > texture(terrainTexture, gl_FragCoord.xy / screenSize).w) {
	// 		discard;
	// 	}
	particleDepth = clipspacePos.z;
}

);