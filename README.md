# Complexity Control Facilitates Reasoning-Based Compositional Generalization in Transformers

This repository contains the code and resources for our research project titled **"Complexity Control Facilitates Reasoning-Based Compositional Generalization in Transformers"**. The project explores enhancing Transformer models' compositional reasoning and out-of-distribution (OOD) generalization through complexity control strategies.

## Project Overview

Transformer models have shown exceptional performance across various tasks but often rely on memorizing input-output mappings, limiting their ability to handle novel combinations of known concepts systematically. Our research introduces complexity control strategies by adjusting parameter initialization scales and weight decay coefficients to steer Transformers from memorization towards learning fundamental rules that support reasoning-based compositional generalization. This approach significantly improves OOD generalization in both synthetic experiments and real-world applications, including image generation and natural language processing.

**Key Contributions:**
- **Complexity Control Strategies:** Adjusting parameter initialization and weight decay to influence model complexity and reasoning capabilities.
- **Distinct Learning Phases:** Categorizing models' learning dynamics into phases with unique internal mechanisms.
- **Enhanced OOD Generalization:** Demonstrating significant improvements in OOD tasks through reasoning-based solutions.
- **Real-World Applications:** Extending findings to image generation tasks, addressing failure modes in prior works.
