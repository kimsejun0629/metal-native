# Security Policy

## Supported Versions

The following versions of MetalNative are currently being supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of MetalNative seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please DO NOT:

- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it has been addressed
- Exploit the vulnerability beyond what is necessary to demonstrate it

### Please DO:

**Report security vulnerabilities via [GitHub Security Advisories](https://github.com/kimsejun/metal-native/security/advisories/new)**

Please include the following information in your report:

1. **Description of the vulnerability**
   - Type of issue (e.g., buffer overflow, memory leak, privilege escalation)
   - Impact assessment (what could an attacker do?)

2. **Steps to reproduce**
   - Detailed step-by-step instructions
   - Proof-of-concept code if available
   - Any special configuration required

3. **Affected versions**
   - Which versions of MetalNative are affected
   - Platform details (macOS version, hardware)

4. **Potential impact**
   - What data or systems could be compromised
   - Severity assessment (Critical, High, Medium, Low)

5. **Suggested fixes** (if you have any)
   - Proposed solutions or mitigations
   - Patches or code changes

### What to expect:

1. **Acknowledgment**: We will acknowledge receipt of your report within **48 hours**.

2. **Initial assessment**: We will provide an initial assessment within **5 business days**, including:
   - Confirmation of the vulnerability
   - Severity rating
   - Estimated timeline for fix

3. **Communication**: We will keep you informed throughout the process:
   - Progress updates every 7-14 days
   - Testing and validation of fixes
   - Planned disclosure timeline

4. **Fix and disclosure**:
   - We aim to address critical vulnerabilities within **30 days**
   - Coordinated disclosure after patch is available
   - Security advisory published on GitHub
   - Credit given to reporter (unless anonymity requested)

5. **Recognition**: 
   - Security researchers will be acknowledged in release notes
   - Hall of fame for significant contributions (if desired)

## Security Best Practices

### For Users

When using MetalNative in your projects:

1. **Keep updated**: Always use the latest stable version with security patches
2. **Validate inputs**: Sanitize and validate all external data before passing to MetalNative
3. **Resource limits**: Set appropriate limits on tensor sizes and memory usage
4. **Sandboxing**: Run untrusted code in sandboxed environments
5. **Monitor resources**: Watch for unusual memory or GPU usage patterns

### For Developers

When contributing to MetalNative:

1. **Input validation**: Always validate buffer sizes, tensor shapes, and indices
2. **Memory safety**: Use RAII patterns and avoid manual memory management
3. **Bounds checking**: Check array accesses and prevent buffer overflows
4. **Integer overflow**: Validate calculations involving sizes and offsets
5. **Resource cleanup**: Ensure proper cleanup even in error paths
6. **Metal API usage**: Follow Apple's security guidelines for Metal
7. **Dependency security**: Keep dependencies updated and monitor advisories

## Security Considerations

### Metal API Security

MetalNative uses Apple's Metal framework. Key security considerations:

1. **GPU Access**: Requires appropriate entitlements and permissions
2. **Shared Memory**: Buffers can be shared between CPU and GPU - validate access patterns
3. **Kernel Security**: Metal shaders run in a sandboxed GPU environment
4. **Resource Limits**: Metal enforces hardware limits on buffer sizes and allocations

### Common Vulnerability Types

We actively monitor for:

1. **Memory Safety**
   - Buffer overflows/underflows
   - Use-after-free
   - Double-free
   - Memory leaks

2. **Integer Vulnerabilities**
   - Integer overflow/underflow
   - Signedness issues
   - Truncation errors

3. **Logic Errors**
   - Incorrect bounds checking
   - Race conditions
   - Improper error handling

4. **Denial of Service**
   - Excessive memory allocation
   - GPU resource exhaustion
   - Infinite loops in kernels

## Known Security Limitations

1. **No Sandboxing**: MetalNative runs with the same privileges as the calling application
2. **GPU Access**: Requires Metal-capable GPU and appropriate system permissions
3. **Memory Isolation**: Shared memory buffers between processes should be carefully managed
4. **Kernel Code**: Custom Metal kernels are not validated or sandboxed by MetalNative

## Security Update Process

1. **Patch Development**: Security fixes are developed in private repositories
2. **Testing**: Comprehensive testing including regression and integration tests
3. **Backporting**: Critical fixes backported to supported versions
4. **Release**: Security releases published with detailed advisories
5. **Notification**: Users notified via GitHub Security Advisories and release notes

## CVE Assignment

For significant vulnerabilities, we will:

1. Request CVE assignment from MITRE or GitHub
2. Include CVE reference in security advisory
3. Update CHANGELOG with CVE information
4. Coordinate with downstream users

## Third-Party Dependencies

MetalNative relies on:

1. **System Frameworks**: Metal, MetalPerformanceShaders, Accelerate (Apple-provided)
2. **Build Dependencies**: CMake, pybind11
3. **Python Runtime**: Python 3.10+

We monitor security advisories for all dependencies and update as needed.

## Security Testing

Our security testing includes:

1. **Static Analysis**: Clang Static Analyzer, clang-tidy
2. **Dynamic Analysis**: AddressSanitizer, ThreadSanitizer
3. **Fuzzing**: Continuous fuzzing of parsing and buffer operations
4. **Code Review**: All code changes reviewed for security implications
5. **Penetration Testing**: Periodic security assessments

## Disclosure Policy

We follow **Coordinated Vulnerability Disclosure**:

1. Private disclosure to maintainers
2. Fix development and testing (typically 30-90 days)
3. Advance notification to major users (if applicable)
4. Public disclosure with patch availability
5. CVE assignment and security advisory

## Questions?

For questions about this security policy, please use [GitHub Security Advisories](https://github.com/kimsejun/metal-native/security/advisories/new) or open a GitHub Discussion.

For general questions (non-security), use GitHub Issues or Discussions.

---

**Last Updated**: February 10, 2024
