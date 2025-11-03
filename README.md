# Visual Reader

is an application for visual learners. 

It is an AI agent which generates visuals, graphs, and diagrams. 

Here is how it works:
You upload a PDF.
You highlight or paste a section of text.
The local Phi-3 model automatically rewrites that section of text it into a more focused prompt for image generation:
“Generate a cinematic view of a lush forest canopy bathed in sunlight, overlay arrows indicating the process of photosynthesis in color.”
That rewritten description is sent to Stable Diffusion Turbo running locally.
The resulting image appears right below the text.

This application runs entirely offline and at no cost. 

![image](https://github.com/brandonlwallace/visual_reader/visual_reader_preview_UI.png) 